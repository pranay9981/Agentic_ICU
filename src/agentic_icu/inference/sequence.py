from __future__ import annotations

import json
import logging
import pickle
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_cudnn_lock = threading.Lock()


class SequenceGRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        gru_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.classifier = nn.Sequential(
            nn.LayerNorm(gru_output_size),
            nn.Linear(gru_output_size, gru_output_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gru_output_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(x)
        final_state = output[:, -1, :]
        return self.classifier(final_state).squeeze(-1)


class SequenceInference:
    def __init__(
        self, model_path: str, metrics_path: str, calibrator_path: Optional[str] = None
    ) -> None:
        self.model_path = Path(model_path)
        self.metrics_path = Path(metrics_path)
        self.calibrator_path = Path(calibrator_path) if calibrator_path else None
        self._model: Optional[SequenceGRU] = None
        self._metrics: Optional[dict] = None
        self._calibrator = None
        self._calibrated_threshold: Optional[float] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self.model_path.exists() and self.metrics_path.exists()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:  # re-check after acquiring
                return
            with self.metrics_path.open("r", encoding="utf-8") as handle:
                metrics = json.load(handle)
            input_size = int(metrics["input_size"])
            arch = metrics.get("architecture", {})
            hidden_size = int(arch.get("hidden_size", 128))
            num_layers = int(arch.get("num_layers", 2))
            dropout = float(arch.get("dropout", 0.2))
            bidirectional = bool(arch.get("bidirectional", False))
            self._metrics = metrics
            self._model = SequenceGRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=bidirectional,
            ).to(self.device)
            # Enforce safe PyTorch loading format
            self._model.load_state_dict(
                torch.load(
                    self.model_path, map_location=self.device, weights_only=True
                )
            )
            self._model.eval()
            if self.calibrator_path and self.calibrator_path.exists():
                with self.calibrator_path.open("rb") as fh:
                    self._calibrator = pickle.load(fh)  # nosec B301
                if self._calibrator is None:
                    raise ValueError(f"Calibrator file loaded as None: {self.calibrator_path}")
                # Re-anchor the threshold to the calibrated probability space so that
                # score comparisons are apples-to-apples.  The threshold stored in
                # the metrics file was chosen on raw (uncalibrated) val probabilities.
                raw_threshold = metrics.get("threshold_selection", {}).get("threshold")
                if raw_threshold is not None:
                    try:
                        self._calibrated_threshold = float(
                            self._calibrator.predict([raw_threshold])[0]
                        )
                    except Exception as exc:
                        logger.warning(
                            "SequenceInference: calibrated threshold computation failed — %s: %s. "
                            "Falling back to raw threshold.",
                            type(exc).__name__, exc,
                        )
                        self._calibrated_threshold = None

    @property
    def metrics(self) -> dict:
        if self._metrics is None:
            self.load()
        return self._metrics or {}

    @property
    def decision_threshold(self) -> float | None:
        threshold_payload = self.metrics.get("threshold_selection", {})
        raw_threshold = threshold_payload.get("threshold")
        if raw_threshold is None:
            return None
        # Return calibrated threshold when the calibrator is loaded so that the
        # threshold lives in the same probability space as the scores we serve.
        if self._calibrator is not None and self._calibrated_threshold is not None:
            return self._calibrated_threshold
        return float(raw_threshold)

    @property
    def calibrated(self) -> bool:
        return self._calibrator is not None

    def predict(self, sequence_tensor) -> float:
        if self._model is None:
            self.load()
        if self._model is None:
            raise RuntimeError("SequenceInference.predict called before model was loaded.")
        array = np.asarray(sequence_tensor, dtype=np.float32)
        tensor = torch.from_numpy(np.expand_dims(array, axis=0)).to(self.device)
        with torch.no_grad():
            raw_score = float(torch.sigmoid(self._model(tensor)).cpu().item())
        if self._calibrator is not None:
            if not hasattr(self._calibrator, 'predict'):
                raise TypeError(f"Calibrator object has no predict() method: {type(self._calibrator)}")
            return float(self._calibrator.predict([raw_score])[0])
        return raw_score

    def temporal_saliency(self, sequence_tensor) -> list[float]:
        """Return per-timestep importance weights via input-gradient saliency.

        Computes |∂sigmoid(output)/∂input| summed over the feature dimension for
        each timestep, then normalises to sum=1 (vanilla-gradient saliency,
        Simonyan et al., 2014).

        Returns:
            List of T floats (one per observation hour) summing to 1.0.
        """
        if self._model is None:
            self.load()
        if self._model is None:
            raise RuntimeError("SequenceInference.temporal_saliency called before model was loaded.")
        array = np.asarray(sequence_tensor, dtype=np.float32)
        tensor = torch.from_numpy(np.expand_dims(array, axis=0)).to(self.device)
        tensor.requires_grad_(True)
        # cuDNN RNN backward requires training mode; disabling cuDNN keeps eval-mode
        # behaviour (no dropout) while allowing gradient flow.
        # Lock guards the global cudnn flag — concurrent saliency calls would race
        # on save/restore and permanently disable cuDNN for all subsequent requests.
        with _cudnn_lock:
            prev_cudnn = torch.backends.cudnn.enabled
            torch.backends.cudnn.enabled = False
            try:
                self._model.zero_grad()
                logit = self._model(tensor)
                score = torch.sigmoid(logit)
                score.backward()
            finally:
                torch.backends.cudnn.enabled = prev_cudnn
                self._model.zero_grad()  # clean parameter .grad buffers after backward
        with torch.no_grad():
            grad = tensor.grad  # shape: (1, T, F)
            if grad is None:
                return [1.0 / array.shape[0]] * array.shape[0]
            per_step = grad.abs().sum(dim=-1).squeeze(0)
            total = per_step.sum()
            if total > 0:
                per_step = per_step / total
        return per_step.cpu().numpy().tolist()

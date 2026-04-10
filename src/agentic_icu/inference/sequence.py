from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


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
    def __init__(self, model_path: str, metrics_path: str, calibrator_path: Optional[str] = None) -> None:
        self.model_path = Path(model_path)
        self.metrics_path = Path(metrics_path)
        self.calibrator_path = Path(calibrator_path) if calibrator_path else None
        self._model: Optional[SequenceGRU] = None
        self._metrics: Optional[dict] = None
        self._calibrator = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def available(self) -> bool:
        return self.model_path.exists() and self.metrics_path.exists()

    def load(self) -> None:
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
        self._model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self._model.eval()
        if self.calibrator_path and self.calibrator_path.exists():
            with self.calibrator_path.open("rb") as fh:
                self._calibrator = pickle.load(fh)

    @property
    def metrics(self) -> dict:
        if self._metrics is None:
            self.load()
        return self._metrics or {}

    @property
    def decision_threshold(self) -> float | None:
        threshold_payload = self.metrics.get("threshold_selection", {})
        threshold = threshold_payload.get("threshold")
        return float(threshold) if threshold is not None else None

    @property
    def calibrated(self) -> bool:
        return self._calibrator is not None

    def predict(self, sequence_tensor) -> float:
        if self._model is None:
            self.load()
        array = np.asarray(sequence_tensor, dtype=np.float32)
        tensor = torch.from_numpy(np.expand_dims(array, axis=0)).to(self.device)
        with torch.no_grad():
            raw_score = float(torch.sigmoid(self._model(tensor)).cpu().item())
        if self._calibrator is not None:
            return float(self._calibrator.predict([raw_score])[0])
        return raw_score

    def temporal_saliency(self, sequence_tensor) -> list[float]:
        """Return per-timestep importance weights via input-gradient saliency.

        Computes |∂sigmoid(output)/∂input| and sums over the feature dimension for
        each timestep, then normalises to sum=1.  This is the standard vanilla-
        gradient saliency method (Simonyan et al., 2014) and works with the
        existing model weights — no retraining required.

        Returns:
            List of T floats (one per observation hour) summing to 1.0.
            Higher values indicate timesteps the model weighted more heavily.
        """
        if self._model is None:
            self.load()
        array = np.asarray(sequence_tensor, dtype=np.float32)
        tensor = torch.from_numpy(np.expand_dims(array, axis=0)).to(self.device)
        tensor.requires_grad_(True)
        # cuDNN RNN backward requires training mode; we disable cuDNN entirely for
        # this pass so dropout remains off (eval mode) and gradients still flow.
        prev_cudnn = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False
        try:
            logit = self._model(tensor)
            score = torch.sigmoid(logit)
            self._model.zero_grad()
            score.backward()
        finally:
            torch.backends.cudnn.enabled = prev_cudnn
        with torch.no_grad():
            grad = tensor.grad  # shape: (1, T, F)
            if grad is None:
                return [1.0 / array.shape[0]] * array.shape[0]
            # Sum absolute gradients over feature dimension → (T,)
            per_step = grad.abs().sum(dim=-1).squeeze(0)
            total = per_step.sum()
            if total > 0:
                per_step = per_step / total
        return per_step.cpu().numpy().tolist()

# Agentic-ICU — Central Monitoring Station

A real-time multi-agent ICU monitoring platform that runs GRU sequence models and XGBoost on live patient vitals and labs to detect sepsis and respiratory failure early, with a clinical-grade dashboard for centralised monitoring.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.135-teal)
![Tests](https://img.shields.io/badge/tests-210%2F210-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

---

## What it does

- **Live central monitoring board** — up to 40,000+ patients, with vitals updating every ~1 second via realistic noise simulation
- **Five AI agents** running per patient evaluation:
  - **Sepsis GRU** — bidirectional GRU sequence model over the 72 h vitals window
  - **Lab XGBoost** — tabular model over 292 engineered features with SHAP explainability
  - **Resp Failure GRU** — dedicated respiratory deterioration sequence model
  - **Ensemble meta-learner** — logistic regression fusing calibrated GRU + XGBoost scores
  - **Signal Quality Agent** — detects motion/artifact and applies partial or full suppression before inference
- **Clinical Reasoner** — fuses all agent scores into `CRITICAL / WATCH / STABLE` with threshold-ratio logic, rationale, and suggested protocol actions
- **Partial SOFA score** — computed from available vitals and labs per evaluation
- **Risk Timeline** — step through up to 72 h of a patient's data to see how risk evolved over time
- **SHAP alert drivers** — top lab features driving each XGBoost prediction
- **Temporal saliency** — GRU attention per observation hour
- **Print / PDF report** — one-click patient report with all scores, SOFA, SHAP drivers, vital trend charts, and the risk timeline
- **Browse 40,000+ patients** — search and add any patient from the dataset to the monitoring board
- **Alert history** — persistent across page reloads (localStorage), with alert annotation on sparklines
- **Keyboard shortcuts** — Escape closes sidebar, ← / → navigates patients, R re-evaluates

---

## Architecture

```
Raw PSV vitals/labs
        │
        ▼
SignalQualityAgent ──► artifact detection + partial/full suppression
        │
        ├──► VitalsAgent  (Sepsis GRU)       → score + temporal saliency
        ├──► LabAgent     (XGBoost)          → score + SHAP contributions
        └──► RespFailure  (Resp GRU)         → respiratory failure score
                    │
                    ▼
            EnsembleInference (logistic meta-learner) → fused calibrated score
                    │
                    ▼
            ClinicalReasoner → alert decision + SOFA + rationale + suggested actions
                    │
                    ▼
         FastAPI REST + WebSocket + Dashboard (HTML/CSS/JS)
```

---

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/pranay9981/Agentic_ICU.git
cd Agentic_ICU
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -e ".[dev]"
```

For production deployment with Gunicorn:
```bash
pip install -e ".[prod]"
```

### 2. Add patient data

Place PSV patient files (PhysioNet Sepsis Challenge format) in:
```
data/raw/p000001.psv
data/raw/p000002.psv
...
```

### 3. Add model artifacts

Place trained model files in `artifacts/`:
```
artifacts/
├── manifest.json                         ← validated on startup
├── xgboost_deterioration_model.json
├── xgboost_calibrator.pkl
├── xgboost_metrics.json
├── xgboost_resp_deterioration_model.json
├── xgboost_resp_calibrator.pkl
├── xgboost_resp_metrics.json
├── sequence_gru_model.pt
├── sequence_gru_calibrator.pkl
├── sequence_gru_metrics.json
├── sequence_resp_gru_model.pt
├── sequence_resp_gru_calibrator.pkl
├── sequence_resp_gru_metrics.json
├── ensemble_meta.pkl
├── ensemble_metrics.json
└── train_statistics.json
```

`manifest.json` format:
```json
{
  "version": "2.0.0",
  "sepsis_gru_input_size": 68,
  "sepsis_xgb_feature_count": 292,
  "resp_gru_input_size": 68,
  "resp_xgb_feature_count": 292
}
```

### 4. Start the server

**Development:**
```bash
# Windows
$env:PYTHONPATH="src"; venv\Scripts\uvicorn agentic_icu.api.main:app --reload

# macOS / Linux
PYTHONPATH=src uvicorn agentic_icu.api.main:app --reload

# Or use the helper script
bash scripts/start_dev.sh
```

**Production (Gunicorn):**
```bash
bash scripts/start_prod.sh
```

Open **http://127.0.0.1:8000** in your browser.

> **API key:** If `settings.api_key` is set in your environment, the dashboard will prompt for `X-API-Key` on first load.

---

## Training your own models

Use the Kaggle training script with the PhysioNet Sepsis Challenge dataset:

```bash
python src/rebuild_training/kaggle_train_deterioration.py \
  --data_dir data/raw \
  --output_dir artifacts \
  --observation_hours 24 \
  --horizon_min_hours 4 \
  --horizon_max_hours 8 \
  --export_sequence_arrays \
  --train_sequence_model \
  --train_resp_failure \
  --sequence_hidden_size 256 \
  --sequence_epochs 30 \
  --sequence_batch_size 128 \
  --sequence_learning_rate 0.0005 \
  --xgb_num_boost_round 2500 \
  --xgb_early_stopping_rounds 100
```

---

## API reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/`                             | Dashboard UI |
| `GET`  | `/health`                       | Model + system health |
| `GET`  | `/runtime-config`               | Active alert policy thresholds |
| `GET`  | `/patients?search=&limit=100`   | Search all patient files |
| `GET`  | `/demo-patient/{id}?max_rows=24`| Load patient observation window |
| `POST` | `/evaluate`                     | Run full multi-agent evaluation |
| `GET`  | `/model-metrics`                | Model performance metrics (AUC, AUPRC, F1, …) |
| `GET`  | `/model-metrics/calibration`    | Isotonic calibration curves for all models |
| `GET`  | `/reports/alert-policy-latest`  | Latest alert policy calibration report |

### Example evaluate request

```bash
curl -X POST http://127.0.0.1:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "p000001",
    "observation_window": [
      {"values": {"HR": 88, "SBP": 118, "MAP": 78, "O2Sat": 97, "Resp": 18, "ICULOS": 1}},
      {"values": {"HR": 102, "SBP": 105, "MAP": 68, "O2Sat": 94, "Resp": 24, "ICULOS": 2}}
    ]
  }'
```

### Example evaluate response (abbreviated)

```json
{
  "patient_id": "p000001",
  "clinical_decision": {
    "alert_type": "Sepsis Early Warning",
    "priority": "high",
    "alert_triggered": true,
    "rationale": "…"
  },
  "vitals_agent":      { "score": 0.921, "risk_band": "high", "threshold_ratio": 1.05 },
  "lab_agent":         { "score": 0.412, "risk_band": "moderate" },
  "resp_failure_agent":{ "score": 0.183, "risk_band": "low" },
  "ensemble_agent":    { "score": 0.874, "risk_band": "high" },
  "sofa": { "total": 4, "interpretation": "moderate", "respiratory": 1, "renal": 2 },
  "signal_quality":    { "signal_valid": true, "suppression_mode": "none" }
}
```

---

## Model performance

All models trained on the **PhysioNet Sepsis Challenge 2019** dataset (40,323 patients, ~6,000 test samples).  
Decision thresholds optimised for F2-score (recall-weighted) to minimise missed events.  
All scores are isotonic-calibrated so they represent true probabilities.

### Sepsis GRU — sequence model (primary deterioration signal)

| Metric | Test |
|--------|------|
| AUC-ROC | **0.923** |
| Avg Precision (AUPRC) | **0.761** |
| Precision (at threshold) | 60.4% |
| Recall | 72.7% |
| F1-Score | 66.0% |
| Brier Score | 0.0356 |
| Decision threshold | 0.0014 (post-calibration) |

Architecture: Bidirectional GRU · hidden=256 · 2 layers · dropout=0.3 · Focal loss · Isotonic calibration  
Input: 68 features × 72 h window

### Respiratory Failure GRU — sequence model

| Metric | Test |
|--------|------|
| AUC-ROC | **0.961** |
| Avg Precision (AUPRC) | **0.921** |
| Precision (at threshold) | 77.4% |
| Recall | 87.8% |
| F1-Score | 82.3% |
| Brier Score | 0.0406 |
| Decision threshold | 0.401 |

### Sepsis XGBoost — tabular model (secondary / SHAP explainability)

| Metric | Test |
|--------|------|
| AUC-ROC | **0.833** |
| Avg Precision (AUPRC) | 0.057 |
| Precision (at threshold) | 5.6% |
| Recall | 53.8% |
| F1-Score | 10.2% |
| Brier Score | 0.0902 |

Features: 292 (vitals + labs + 13 composite clinical features: qSOFA, shock index, SpO₂/FiO₂ ratio, pulse pressure, MAP, …)

> **Note on XGBoost AUPRC:** The sepsis label is highly imbalanced (~1% of hourly observations). The model's AUC of 0.833 and recall of 54% are clinically useful as a second-signal trigger; SHAP contributions are the primary value of the tabular model.

### Respiratory Failure XGBoost — tabular model (context only)

| Metric | Test |
|--------|------|
| AUC-ROC | 0.686 |
| Avg Precision (AUPRC) | 0.011 |

> Used as a contextual signal only. The Resp GRU (AUC 0.961) is the primary respiratory failure predictor.

### Ensemble meta-learner (GRU + XGBoost)

Logistic regression fusing the calibrated sepsis GRU and XGBoost scores. Trained on held-out validation predictions. Improves discrimination by combining temporal patterns (GRU) with lab context (XGBoost).

---

## Alert policy

Configured in `configs/runtime_alert_policy.json`:

| Level | Trigger condition |
|-------|------------------|
| **CRITICAL (extreme)** | Sepsis GRU ≥ 0.88 |
| **CRITICAL (supported)** | Sepsis GRU ≥ 0.80 AND XGBoost ≥ 0.25 |
| **CRITICAL (ensemble)** | Ensemble score ≥ 0.80 |
| **WATCH** | Sepsis GRU ≥ 0.55 OR XGBoost ≥ 0.40 OR Resp GRU ≥ 0.55 |
| **STABLE** | All scores below watch thresholds |

Partial suppression (0.7×) is applied to all scores when the Signal Quality Agent detects artifact.  
Full suppression short-circuits inference entirely and returns `Suppressed Artifact`.

---

## Partial SOFA score

Computed per evaluation from available PSV features. Components:

| Component | Source | Notes |
|-----------|--------|-------|
| Respiratory | SpO₂/FiO₂ ratio | FiO₂ estimated from flow rate if not measured |
| Coagulation | Platelets | Direct measurement |
| Liver | Bilirubin | Direct measurement |
| Cardiovascular | MAP | Vasopressor data not available in dataset |
| CNS | — | GCS not included in PSV format |
| Renal | Creatinine + urine output | |

SOFA interpretation: 0–1 = low · 2–5 = moderate · 6–9 = high · ≥10 = critical

---

## Running tests

```bash
# Windows
$env:PYTHONPATH="src"; venv\Scripts\python -m pytest tests/ -v

# macOS / Linux
PYTHONPATH=src python -m pytest tests/ -v
```

210 tests covering: agents, inference models, ensemble, SOFA, API endpoints, contracts, alert policy tools, and preprocessing.

---

## Project structure

```
Agentic-ICU/
├── src/
│   ├── agentic_icu/
│   │   ├── agents/          # SignalQuality, Vitals, Lab, RespFailure, Reasoner
│   │   ├── api/
│   │   │   ├── main.py      # FastAPI app, WebSocket, rate limiting, health checks
│   │   │   └── static/      # Dashboard (index.html, app.js, app.css)
│   │   ├── inference/       # GRU sequence predictor, XGBoost tabular, Ensemble, SHAP explainer
│   │   ├── orchestration/   # Multi-agent workflow
│   │   ├── preprocessing/   # Windowing + 292-feature engineering
│   │   ├── tools/           # SOFA calculator, alert policy evaluator
│   │   └── domain/          # Pydantic contracts (request/response schemas)
│   └── rebuild_training/    # Kaggle training scripts
├── artifacts/               # Trained model files + manifest.json (add manually)
├── configs/                 # Alert policy JSON configs
├── data/raw/                # Patient PSV files (not in repo — add your own)
├── scripts/                 # start_dev.sh, start_prod.sh
└── tests/                   # 210 tests
```

---

## Dataset

This project uses the **PhysioNet Computing in Cardiology Challenge 2019** dataset (Sepsis Early Prediction).  
Download from: https://physionet.org/content/challenge-2019/1.0.0/

Patient data is **not included** in this repository.

---

## Known limitations

- **XGBoost AUPRC** (0.057 sepsis, 0.011 resp) is low due to high class imbalance (~1% of hourly rows are positive). The XGBoost is used primarily for SHAP explanations and as a corroborating signal, not as a standalone predictor.
- **Resp XGBoost** uses proxy labels (SpO₂ < 90% sustained), not true clinical respiratory failure diagnoses. The Resp GRU is the primary signal.
- **SOFA cardiovascular component** is MAP-only — vasopressor data is not in the PSV format.
- **WebSocket broadcasts** are in-process only. In a multi-worker Gunicorn deployment, each worker has its own connection pool. Redis pub/sub would be required for true cross-worker broadcasting.
- **No HIPAA audit logging or at-rest encryption** — this is a research prototype, not a production clinical system.

---

## License

MIT

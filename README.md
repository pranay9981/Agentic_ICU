# Agentic-ICU — Central Monitoring Station

A real-time multi-agent ICU monitoring dashboard that runs GRU sequence models and XGBoost on live patient vitals/labs to detect sepsis and respiratory failure early.

![Dashboard](https://img.shields.io/badge/status-active-brightgreen) ![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.135-teal) ![License](https://img.shields.io/badge/license-MIT-green)

---

## What it does

- **Live central monitoring board** — all ICU patients visible simultaneously, vitals update every ~1 second with realistic noise simulation
- **Three AI agents** running in parallel per patient:
  - **Sepsis GRU** — sequence model over 24h vitals window
  - **Lab XGBoost** — tabular model with SHAP explanations
  - **Resp Failure GRU** — respiratory deterioration detection
- **Clinical Reasoner** — fuses all three scores into a `CRITICAL / WATCH / STABLE` decision with suggested actions
- **Signal Quality Agent** — detects and suppresses artifact before inference
- **Per-frame timeline** — step through 24 hours of a patient's data to see how risk evolved
- **Browse 40k+ patients** — search and add any patient from the dataset

---

## Architecture

```
Raw PSV vitals/labs
        │
        ▼
SignalQualityAgent ──► artifact detection + suppression
        │
        ├──► VitalsAgent (GRU)      → sepsis score + temporal saliency
        ├──► LabAgent (XGBoost)     → sepsis score + SHAP contributions
        └──► RespFailureAgent (GRU) → respiratory failure score
                    │
                    ▼
            ClinicalReasoner → alert decision + rationale + suggested actions
                    │
                    ▼
             FastAPI + Dashboard
```

---

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/pranay9981/Agentic-ICU.git
cd Agentic-ICU
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

> **GPU (CUDA):** Replace the torch line in requirements with the matching CUDA wheel:
> ```bash
> pip install torch==2.10.0+cu130 --index-url https://download.pytorch.org/whl/cu130
> ```

### 2. Add patient data

Place your PSV patient files (PhysioNet Sepsis Challenge format) in:
```
data/raw/p000001.psv
data/raw/p000002.psv
...
```

### 3. Add model artifacts

Place trained model files in `artifacts/`:
```
artifacts/
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
└── train_statistics.json
```

### 4. Start the server

```bash
# Windows
$env:PYTHONPATH="src"; venv\Scripts\uvicorn agentic_icu.api.main:app --reload

# macOS / Linux
PYTHONPATH=src uvicorn agentic_icu.api.main:app --reload
```

Open **http://127.0.0.1:8000** in your browser.

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
| `GET` | `/` | Dashboard UI |
| `GET` | `/health` | Model + system health |
| `GET` | `/runtime-config` | Active alert policy thresholds |
| `GET` | `/patients?search=&limit=100` | Search all patient files |
| `GET` | `/demo-patient/{id}?max_rows=24` | Load patient observation window |
| `POST` | `/evaluate` | Run full multi-agent evaluation |
| `GET` | `/reports/alert-policy-latest` | Latest calibration report |

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

---

## Model performance

All models trained on the **PhysioNet Sepsis Challenge 2019** dataset (40,323 patients, ~6,000 test samples).  
Decision thresholds optimised for F2-score (recall-weighted) to minimise missed events.

### Sepsis GRU — sequence model (primary)

| Metric | Validation | Test |
|--------|-----------|------|
| AUC-ROC | 0.928 | **0.923** |
| Avg Precision (AUPRC) | 0.770 | **0.761** |
| Precision (at threshold) | 59.7% | 60.4% |
| Recall | 75.0% | 72.7% |
| F1-Score | 66.5% | 66.0% |
| Brier Score | 0.0356 | 0.0356 |
| Decision threshold | — | 0.0014 |

Architecture: Bidirectional GRU · hidden=256 · 2 layers · dropout=0.3 · Focal loss · Isotonic calibration

### Respiratory Failure GRU — sequence model (primary)

| Metric | Validation | Test |
|--------|-----------|------|
| AUC-ROC | 0.967 | **0.961** |
| Avg Precision (AUPRC) | 0.927 | **0.921** |
| Precision (at threshold) | 77.6% | 77.4% |
| Recall | 89.2% | 87.8% |
| F1-Score | 83.0% | 82.3% |
| Brier Score | 0.0370 | 0.0406 |
| Decision threshold | — | 0.401 |

Architecture: Bidirectional GRU · hidden=256 · 2 layers · dropout=0.3 · Focal loss · Isotonic calibration

### Sepsis XGBoost — tabular model (secondary / SHAP explainability)

| Metric | Validation | Test |
|--------|-----------|------|
| AUC-ROC | 0.838 | **0.833** |
| Avg Precision (AUPRC) | 0.054 | **0.057** |
| Precision (at threshold) | 6.5% | 5.6% |
| Recall | 53.3% | 53.8% |
| F1-Score | 11.6% | 10.2% |
| Brier Score | 0.0845 | 0.0902 |

Features: 292 (vitals + labs + 13 composite clinical features including qSOFA, shock index, SpO₂/FiO₂ ratio)  
Top drivers: ICULOS, FiO₂ observation gap, PaCO₂ gap, EtCO₂ fraction, HR×ICU-LOS

> **Note on low AUPRC:** The sepsis label is highly imbalanced (~1% of hourly observations). AUPRC reflects this base rate — the model's AUC of 0.833 and recall of 54% remain clinically useful for triggering the GRU re-evaluation chain.

### Respiratory Failure XGBoost — tabular model (secondary / context)

| Metric | Validation | Test |
|--------|-----------|------|
| AUC-ROC | 0.720 | **0.686** |
| Avg Precision (AUPRC) | 0.012 | **0.011** |
| Brier Score | 0.0615 | 0.0630 |

> Used as a contextual signal only. The resp GRU (AUC 0.961) is the primary respiratory failure predictor.

---

## Alert policy

Configured in `configs/runtime_alert_policy.json`:

| Level | Trigger condition |
|-------|------------------|
| **CRITICAL** | Sepsis GRU ≥ 0.88 OR (GRU ≥ 0.80 AND XGBoost ≥ 0.25) |
| **WATCH** | Sepsis GRU ≥ 0.55 OR XGBoost ≥ 0.40 OR Resp GRU ≥ 0.55 |
| **STABLE** | All scores below watch thresholds |

---

## Running tests

```bash
PYTHONPATH=src venv\Scripts\python -m unittest discover -s tests -v
```

21 integration tests covering stable, watch, and critical patient paths plus signal quality, SHAP, and temporal saliency.

---

## Project structure

```
Agentic-ICU/
├── src/
│   ├── agentic_icu/
│   │   ├── agents/          # SignalQuality, Vitals, Lab, RespFailure, Reasoner
│   │   ├── api/             # FastAPI app + static dashboard (HTML/CSS/JS)
│   │   ├── inference/       # GRU sequence predictor, XGBoost tabular predictor, SHAP explainer
│   │   ├── orchestration/   # Multi-agent workflow
│   │   ├── preprocessing/   # Windowing + feature engineering
│   │   └── domain/          # Pydantic contracts
│   └── rebuild_training/    # Kaggle training script
├── artifacts/               # Trained model files (not in git — add manually)
├── configs/                 # Alert policy JSON configs
├── data/raw/                # Patient PSV files (not in git — add your own)
├── tests/                   # Integration tests
└── requirements.txt
```

---

## Dataset

This project uses the **PhysioNet Computing in Cardiology Challenge 2019** dataset (Sepsis Early Prediction).  
Download from: https://physionet.org/content/challenge-2019/1.0.0/

Patient data is **not included** in this repository.

---

## License

MIT

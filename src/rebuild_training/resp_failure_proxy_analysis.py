"""Respiratory failure proxy label feasibility analysis.

Scans all 40k PhysioNet PSV files and evaluates three candidate proxy
definitions.  Reports positive row rate, positive patient rate, median
hours before SepsisLabel onset, and overlap statistics so we can decide
whether the proxy is viable for training.

Run from repo root:
    python src/rebuild_training/resp_failure_proxy_analysis.py
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"

# ── Proxy definitions ─────────────────────────────────────────────────────────
# Each returns True/False per row given the carry-forward value dict.
# Labels are then sustained-window smoothed (SUSTAIN_HOURS consecutive True rows).

SUSTAIN_HOURS = 2          # criteria must hold for this many consecutive hours
LOOKAHEAD_HOURS = 8        # label at hour t if criteria met in t+1..t+LOOKAHEAD

def proxy_tier1(row: dict) -> bool:
    """Hypoxemic failure: low SpO2 + high Resp OR low SpO2/FiO2 ratio."""
    spo2 = row.get("O2Sat")
    resp = row.get("Resp")
    fio2 = row.get("FiO2")

    # Arm A: SpO2 < 90 and Resp > 25
    if spo2 is not None and resp is not None:
        if spo2 < 90 and resp > 25:
            return True

    # Arm B: SpO2/FiO2 ratio < 235 (proxy for PaO2/FiO2 < 300 — Berlin mild ARDS)
    if spo2 is not None and fio2 is not None and fio2 > 0:
        if (spo2 / fio2) < 235:
            return True

    return False


def proxy_tier2(row: dict) -> bool:
    """Ventilatory + hypoxemic: SpO2 < 93 + Resp > 22 + acidosis or hypercapnia."""
    spo2 = row.get("O2Sat")
    resp = row.get("Resp")
    ph   = row.get("pH")
    pco2 = row.get("PaCO2")

    if spo2 is None or resp is None:
        return False

    if not (spo2 < 93 and resp > 22):
        return False

    # Need at least one acid-base marker
    if ph is not None and ph < 7.35:
        return True
    if pco2 is not None and pco2 > 50:
        return True

    return False


def proxy_combined(row: dict) -> bool:
    """Either tier."""
    return proxy_tier1(row) or proxy_tier2(row)


PROXIES = {
    "tier1_hypoxemic":   proxy_tier1,
    "tier2_ventilatory": proxy_tier2,
    "combined":          proxy_combined,
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_psv(path: Path) -> list[dict]:
    """Return list of row dicts with float values (NaN stripped)."""
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="|")
        for raw in reader:
            row: dict = {}
            for k, v in raw.items():
                try:
                    f = float(v)
                    import math
                    if math.isfinite(f):
                        row[k] = f
                except (TypeError, ValueError):
                    pass
            rows.append(row)
    return rows


def carry_forward(rows: list[dict]) -> list[dict]:
    """Return rows with missing numeric values filled from previous row."""
    carried: list[dict] = []
    state: dict = {}
    for row in rows:
        state = {**state, **row}
        carried.append(dict(state))
    return carried


def sustained_flag(raw_flags: list[bool], window: int) -> list[bool]:
    """True at index i if raw_flags[i-window+1 .. i] are all True."""
    n = len(raw_flags)
    result = [False] * n
    count = 0
    for i, flag in enumerate(raw_flags):
        count = (count + 1) if flag else 0
        result[i] = count >= window
    return result


def lookahead_label(sustained: list[bool], horizon: int) -> list[bool]:
    """Label row t=1 if any of t+1..t+horizon is sustained=True."""
    n = len(sustained)
    labels = [False] * n
    for i in range(n):
        for j in range(i + 1, min(i + horizon + 1, n)):
            if sustained[j]:
                labels[i] = True
                break
    return labels


# ── Main analysis ─────────────────────────────────────────────────────────────

def analyse() -> None:
    import math

    psv_files = sorted(RAW_DIR.glob("*.psv"))
    total_files = len(psv_files)
    print(f"Scanning {total_files} patient files in {RAW_DIR} ...\n")

    stats: dict[str, dict] = {
        name: {
            "total_rows": 0,
            "pos_rows": 0,
            "pos_patients": 0,
            "total_patients": 0,
            # hours before SepsisLabel onset for patients with both labels
            "lead_hours": [],
            # patients with resp proxy but no SepsisLabel
            "resp_only_patients": 0,
            # patients with SepsisLabel but no resp proxy
            "sepsis_only_patients": 0,
            # patients with both
            "overlap_patients": 0,
        }
        for name in PROXIES
    }

    for idx, path in enumerate(psv_files):
        if idx % 5000 == 0:
            print(f"  {idx}/{total_files} ...", flush=True)

        rows = parse_psv(path)
        if not rows:
            continue
        filled = carry_forward(rows)

        sepsis_labels = [bool(r.get("SepsisLabel", 0)) for r in filled]
        has_sepsis = any(sepsis_labels)
        sepsis_onset = next((i for i, v in enumerate(sepsis_labels) if v), None)

        for name, fn in PROXIES.items():
            s = stats[name]
            s["total_patients"] += 1
            s["total_rows"] += len(filled)

            raw_flags = [fn(r) for r in filled]
            sus = sustained_flag(raw_flags, SUSTAIN_HOURS)
            labels = lookahead_label(sus, LOOKAHEAD_HOURS)

            pos_rows = sum(labels)
            has_resp = pos_rows > 0

            s["pos_rows"] += pos_rows
            if has_resp:
                s["pos_patients"] += 1

            # Lead-time vs SepsisLabel
            if has_sepsis and has_resp and sepsis_onset is not None:
                resp_onset = next((i for i, v in enumerate(labels) if v), None)
                if resp_onset is not None:
                    lead = sepsis_onset - resp_onset
                    s["lead_hours"].append(lead)

            # Overlap breakdown
            if has_resp and has_sepsis:
                s["overlap_patients"] += 1
            elif has_resp and not has_sepsis:
                s["resp_only_patients"] += 1
            elif not has_resp and has_sepsis:
                s["sepsis_only_patients"] += 1

    # ── Print results ──────────────────────────────────────────────────────────
    sep = "-" * 72
    print(f"\n{sep}")
    print("RESPIRATORY FAILURE PROXY LABEL FEASIBILITY REPORT")
    print(sep)

    for name, s in stats.items():
        total_rows = s["total_rows"]
        pos_rows   = s["pos_rows"]
        total_pat  = s["total_patients"]
        pos_pat    = s["pos_patients"]
        row_rate   = 100 * pos_rows   / total_rows  if total_rows else 0
        pat_rate   = 100 * pos_pat    / total_pat   if total_pat  else 0

        leads = s["lead_hours"]
        lead_median = sorted(leads)[len(leads) // 2] if leads else None
        lead_mean   = sum(leads) / len(leads) if leads else None

        print(f"\n-- {name.upper()} --")
        print(f"  Positive rows:     {pos_rows:>7,} / {total_rows:>7,}  ({row_rate:.2f}%)")
        print(f"  Positive patients: {pos_pat:>7,} / {total_pat:>7,}  ({pat_rate:.2f}%)")
        print(f"  Overlap (both labels): {s['overlap_patients']:,}")
        print(f"  Resp-only patients:    {s['resp_only_patients']:,}  (independent signal)")
        print(f"  Sepsis-only patients:  {s['sepsis_only_patients']:,}")
        if leads:
            print(f"  Lead time vs SepsisLabel: median={lead_median:+.0f}h, mean={lead_mean:+.1f}h")
            print(f"    (positive = resp proxy fires BEFORE sepsis; negative = after)")
        else:
            print(f"  No co-occurring patients for lead-time analysis.")

        # Viability verdict
        viable = 3.0 <= row_rate <= 20.0 and pos_pat >= 500
        verdict = "VIABLE" if viable else "MARGINAL -- adjust thresholds"
        print(f"\n  VERDICT: {verdict}")
        print(f"    Target: 3-20% row rate, >=500 positive patients")

    print(f"\n{sep}")
    print("RECOMMENDATION")
    print(sep)
    print("""
  Choose the proxy with:
    1. Row rate between 3-20%
    2. Highest resp_only_patients count (most independent signal from SepsisLabel)
    3. Positive lead time vs SepsisLabel (fires before, not after)

  If combined proxy is viable, use it -- it captures both hypoxemic and
  ventilatory failure patterns and gives the model richer signal.

  Next step: add build_resp_failure_label() to kaggle_train_deterioration.py
  using the chosen proxy definition, then retrain on Kaggle.
""")


if __name__ == "__main__":
    analyse()

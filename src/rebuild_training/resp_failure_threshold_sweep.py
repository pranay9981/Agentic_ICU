"""Threshold sweep for the O2Sat/FiO2 respiratory failure proxy.

Sweeps O2Sat/FiO2 cutoffs from 120 to 250 and O2Sat+Resp combinations
to find the threshold that lands at 5-8% positive row rate.

Run from repo root:
    python src/rebuild_training/resp_failure_threshold_sweep.py
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"

SUSTAIN_HOURS = 2
LOOKAHEAD_HOURS = 8

# Sweep grid
SF_RATIO_CUTOFFS = list(range(120, 255, 15))  # O2Sat/FiO2 thresholds
SPO2_CUTOFFS = [85, 87, 88, 90]  # SpO2 arm thresholds
RESP_CUTOFFS = [26, 28, 30]  # Resp arm thresholds


def parse_psv(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="|")
        for raw in reader:
            row: dict = {}
            for k, v in raw.items():
                try:
                    f = float(v)
                    if math.isfinite(f):
                        row[k] = f
                except (TypeError, ValueError):
                    pass
            rows.append(row)
    return rows


def carry_forward(rows: list[dict]) -> list[dict]:
    carried: list[dict] = []
    state: dict = {}
    for row in rows:
        state = {**state, **row}
        carried.append(dict(state))
    return carried


def sustained_flag(raw_flags: list[bool], window: int) -> list[bool]:
    result = [False] * len(raw_flags)
    count = 0
    for i, flag in enumerate(raw_flags):
        count = (count + 1) if flag else 0
        result[i] = count >= window
    return result


def lookahead_label(sustained: list[bool], horizon: int) -> list[bool]:
    n = len(sustained)
    labels = [False] * n
    for i in range(n):
        for j in range(i + 1, min(i + horizon + 1, n)):
            if sustained[j]:
                labels[i] = True
                break
    return labels


def count_positive(filled: list[dict], proxy_fn) -> int:
    raw = [proxy_fn(r) for r in filled]
    sus = sustained_flag(raw, SUSTAIN_HOURS)
    lab = lookahead_label(sus, LOOKAHEAD_HOURS)
    return sum(lab)


def analyse() -> None:
    psv_files = sorted(RAW_DIR.glob("*.psv"))
    total_files = len(psv_files)
    print(f"Scanning {total_files} patient files ...\n")

    # Accumulate per-candidate stats
    # candidates keyed by label string
    candidates: dict[str, dict] = {}

    # Build candidate list
    # Group A: pure O2Sat/FiO2 ratio
    for cutoff in SF_RATIO_CUTOFFS:
        key = f"SF<{cutoff}"
        candidates[key] = {
            "pos_rows": 0,
            "pos_patients": 0,
            "total_rows": 0,
            "total_patients": 0,
        }

    # Group B: SpO2 + Resp combination (stricter than tier1)
    for spo2_cut in SPO2_CUTOFFS:
        for resp_cut in RESP_CUTOFFS:
            key = f"SpO2<{spo2_cut}+Resp>{resp_cut}"
            candidates[key] = {
                "pos_rows": 0,
                "pos_patients": 0,
                "total_rows": 0,
                "total_patients": 0,
            }

    # Group C: SF ratio OR SpO2+Resp (combined tighter)
    for sf_cut in [150, 165, 180, 200]:
        for spo2_cut in [87, 88]:
            for resp_cut in [28, 30]:
                key = f"SF<{sf_cut}|SpO2<{spo2_cut}+Resp>{resp_cut}"
                candidates[key] = {
                    "pos_rows": 0,
                    "pos_patients": 0,
                    "total_rows": 0,
                    "total_patients": 0,
                }

    total_rows_global = 0

    for idx, path in enumerate(psv_files):
        if idx % 5000 == 0:
            print(f"  {idx}/{total_files} ...", flush=True)

        rows = parse_psv(path)
        if not rows:
            continue
        filled = carry_forward(rows)
        n = len(filled)
        total_rows_global += n

        # Build feature arrays once per patient
        sf_vals: list[Optional[float]] = []
        spo2_vals: list[Optional[float]] = []
        resp_vals: list[Optional[float]] = []
        for r in filled:
            spo2 = r.get("O2Sat")
            fio2 = r.get("FiO2")
            resp = r.get("Resp")
            sf = (
                (spo2 / fio2)
                if (spo2 is not None and fio2 is not None and fio2 > 0)
                else None
            )
            sf_vals.append(sf)
            spo2_vals.append(spo2)
            resp_vals.append(resp)

        # Group A
        for cutoff in SF_RATIO_CUTOFFS:
            key = f"SF<{cutoff}"
            raw = [(sf is not None and sf < cutoff) for sf in sf_vals]
            sus = sustained_flag(raw, SUSTAIN_HOURS)
            lab = lookahead_label(sus, LOOKAHEAD_HOURS)
            pos = sum(lab)
            candidates[key]["total_rows"] += n
            candidates[key]["total_patients"] += 1
            candidates[key]["pos_rows"] += pos
            if pos > 0:
                candidates[key]["pos_patients"] += 1

        # Group B
        for spo2_cut in SPO2_CUTOFFS:
            for resp_cut in RESP_CUTOFFS:
                key = f"SpO2<{spo2_cut}+Resp>{resp_cut}"
                def _check_group_b(i: int) -> bool:
                    s = spo2_vals[i]
                    r = resp_vals[i]
                    return s is not None and s < spo2_cut and r is not None and r > resp_cut

                raw = [_check_group_b(i) for i in range(n)]
                sus = sustained_flag(raw, SUSTAIN_HOURS)
                lab = lookahead_label(sus, LOOKAHEAD_HOURS)
                pos = sum(lab)
                candidates[key]["total_rows"] += n
                candidates[key]["total_patients"] += 1
                candidates[key]["pos_rows"] += pos
                if pos > 0:
                    candidates[key]["pos_patients"] += 1

        # Group C
        for sf_cut in [150, 165, 180, 200]:
            for spo2_cut in [87, 88]:
                for resp_cut in [28, 30]:
                    key = f"SF<{sf_cut}|SpO2<{spo2_cut}+Resp>{resp_cut}"
                    def _check_group_c(i: int) -> bool:
                        sf = sf_vals[i]
                        s = spo2_vals[i]
                        r = resp_vals[i]
                        return (sf is not None and sf < sf_cut) or (
                            s is not None and s < spo2_cut and r is not None and r > resp_cut
                        )

                    raw = [_check_group_c(i) for i in range(n)]
                    sus = sustained_flag(raw, SUSTAIN_HOURS)
                    lab = lookahead_label(sus, LOOKAHEAD_HOURS)
                    pos = sum(lab)
                    candidates[key]["total_rows"] += n
                    candidates[key]["total_patients"] += 1
                    candidates[key]["pos_rows"] += pos
                    if pos > 0:
                        candidates[key]["pos_patients"] += 1

    # ── Print results ──────────────────────────────────────────────────────────
    sep = "-" * 72
    print(f"\n{sep}")
    print("THRESHOLD SWEEP RESULTS")
    print(f"{sep}")
    print(f"{'Candidate':<40} {'Row%':>6}  {'Pat%':>6}  {'PosPat':>7}  {'Status'}")
    print(sep)

    TARGET_LOW, TARGET_HIGH = 3.0, 10.0
    viable_candidates = []

    for key, s in sorted(candidates.items(), key=lambda x: x[1]["pos_rows"]):
        total_rows = s["total_rows"]
        pos_rows = s["pos_rows"]
        total_pat = s["total_patients"]
        pos_pat = s["pos_patients"]
        row_rate = 100 * pos_rows / total_rows if total_rows else 0
        pat_rate = 100 * pos_pat / total_pat if total_pat else 0
        viable = TARGET_LOW <= row_rate <= TARGET_HIGH and pos_pat >= 500

        status = (
            "VIABLE"
            if viable
            else ("too broad" if row_rate > TARGET_HIGH else "too narrow")
        )
        print(
            f"{key:<40} {row_rate:>5.2f}%  {pat_rate:>5.2f}%  {pos_pat:>7,}  {status}"
        )

        if viable:
            viable_candidates.append((key, row_rate, pos_pat))

    print(f"\n{sep}")
    print("VIABLE CANDIDATES (3-10% row rate, >=500 positive patients)")
    print(sep)
    if viable_candidates:
        for key, row_rate, pos_pat in sorted(viable_candidates, key=lambda x: x[1]):
            print(f"  {key:<40} row_rate={row_rate:.2f}%  pos_patients={pos_pat:,}")

        # Pick the one closest to 6% row rate (middle of target band)
        best = min(viable_candidates, key=lambda x: abs(x[1] - 6.0))
        print(f"\n  RECOMMENDED: {best[0]}")
        print(f"    Row rate {best[1]:.2f}% -- closest to 6% target midpoint")
        print(f"    ~{best[2]:,} positive patients for training")
    else:
        print("  No candidates in target band. Consider relaxing SUSTAIN_HOURS to 1.")

    print(f"\n{sep}")
    print("NEXT STEP")
    print(sep)
    print("""
  Use the recommended proxy definition in build_resp_failure_label()
  inside kaggle_train_deterioration.py, then retrain on Kaggle.
""")


if __name__ == "__main__":
    analyse()

from __future__ import annotations

from agentic_icu.domain.contracts import AgentLogEntry, SignalQualityResult

# ── Physiological plausibility bounds ────────────────────────────────────────
_BOUNDS: dict[str, tuple[float, float]] = {
    "HR":    (15.0,  300.0),
    "Pulse": (15.0,  300.0),
    "SBP":   (30.0,  300.0),
    "DBP":   (10.0,  200.0),
    "MAP":   (15.0,  200.0),
    "Resp":  ( 3.0,   60.0),
    "Temp":  (26.0,   43.0),
    "O2Sat": (50.0,  100.0),
}

_HARD_BLOCK_FEATURES = {"HR", "Pulse", "SBP", "O2Sat"}

_FLATLINE_MIN_ROWS = 5         # consecutive identical values → flatline
_TREND_WINDOW = 6              # rows used for window-based trend checks
_HR_JUMP_THRESHOLD = 40.0      # BPM change in one hour
_SBP_DROP_THRESHOLD = 35.0     # mmHg drop with no HR response
_SPO2_PARADOX_SPO2 = 80.0      # SpO2 below this …
_SPO2_PARADOX_HR_DELTA = 10.0  # … but HR barely moved (probe-off sign)
_PERSISTENT_LOW_SPO2 = 85.0    # SpO2 threshold for multi-row probe-off check


def _get(row: dict[str, float], *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        # Exclude bools — Python bool is a subtype of int; True/False as vitals
        # would pass isinstance(value, (int, float)) and produce 0/1 bpm/mmHg.
        if value is not None and not isinstance(value, bool) and isinstance(value, (int, float)):
            return float(value)
    return None


class SignalQualityAgent:
    def evaluate(self, window_values: list[dict[str, float]]) -> tuple[SignalQualityResult, list[AgentLogEntry]]:
        logs: list[AgentLogEntry] = []

        if len(window_values) < 2:
            result = SignalQualityResult(signal_valid=True)
            logs.append(AgentLogEntry(
                agent="Signal Quality",
                message="Insufficient history for artifact checks; stream accepted.",
            ))
            return result, logs

        current = window_values[-1]
        previous = window_values[-2]

        # ── 1. HR impossible range ─────────────────────────────────────────
        curr_hr = _get(current, "HR", "Pulse")
        if curr_hr is not None:
            lo, hi = _BOUNDS["HR"]
            if not (lo <= curr_hr <= hi):
                result = SignalQualityResult(
                    signal_valid=False,
                    artifact_type="impossible_hr",
                    artifact_confidence=0.96,
                    suppression_recommendation=True,
                    artifact_affected_features=["HR", "Pulse"],
                    suppression_mode="full",
                )
                logs.append(AgentLogEntry(
                    agent="Signal Quality",
                    message=f"Suppressed: HR {curr_hr:.0f} outside physiological range [{lo:.0f}, {hi:.0f}].",
                ))
                return result, logs

        # ── 2. SBP impossible range ────────────────────────────────────────
        curr_sbp = _get(current, "SBP")
        if curr_sbp is not None:
            lo, hi = _BOUNDS["SBP"]
            if not (lo <= curr_sbp <= hi):
                result = SignalQualityResult(
                    signal_valid=False,
                    artifact_type="impossible_sbp",
                    artifact_confidence=0.96,
                    suppression_recommendation=True,
                    artifact_affected_features=["SBP"],
                    suppression_mode="full",
                )
                logs.append(AgentLogEntry(
                    agent="Signal Quality",
                    message=f"Suppressed: SBP {curr_sbp:.0f} outside physiological range [{lo:.0f}, {hi:.0f}].",
                ))
                return result, logs

        # ── 3. SpO2 impossible range ───────────────────────────────────────
        curr_spo2 = _get(current, "O2Sat")
        if curr_spo2 is not None:
            lo, hi = _BOUNDS["O2Sat"]
            if not (lo <= curr_spo2 <= hi):
                result = SignalQualityResult(
                    signal_valid=False,
                    artifact_type="invalid_spo2",
                    artifact_confidence=0.95,
                    suppression_recommendation=True,
                    artifact_affected_features=["O2Sat"],
                    suppression_mode="full",
                )
                logs.append(AgentLogEntry(
                    agent="Signal Quality",
                    message=f"Suppressed: SpO2 {curr_spo2:.0f}% outside physiological range.",
                ))
                return result, logs

        # ── 4. Improbable HR jump ──────────────────────────────────────────
        prev_hr = _get(previous, "HR", "Pulse")
        if curr_hr is not None and prev_hr is not None and abs(curr_hr - prev_hr) > _HR_JUMP_THRESHOLD:
            result = SignalQualityResult(
                signal_valid=False,
                artifact_type="improbable_hr_jump",
                artifact_confidence=0.92,
                suppression_recommendation=True,
                artifact_affected_features=["HR", "Pulse"],
                suppression_mode="full",
            )
            logs.append(AgentLogEntry(
                agent="Signal Quality",
                message="Suppressed: improbable heart-rate jump between consecutive readings.",
            ))
            return result, logs

        # ── 5. Isolated BP drop (no HR corroboration) ─────────────────────
        prev_sbp = _get(previous, "SBP")
        if (
            curr_sbp is not None
            and prev_sbp is not None
            and (prev_sbp - curr_sbp) > _SBP_DROP_THRESHOLD
            and curr_hr is not None
            and prev_hr is not None
            and abs(curr_hr - prev_hr) < 5
        ):
            result = SignalQualityResult(
                signal_valid=False,
                artifact_type="isolated_bp_drop",
                artifact_confidence=0.88,
                suppression_recommendation=True,
                artifact_affected_features=["SBP", "DBP", "MAP"],
                suppression_mode="full",
            )
            logs.append(AgentLogEntry(
                agent="Signal Quality",
                message="Suppressed: isolated BP collapse without corroborating physiology.",
            ))
            return result, logs

        # ── 6. SBP < DBP pressure inversion ──────────────────────────────
        curr_dbp = _get(current, "DBP")
        if curr_sbp is not None and curr_dbp is not None and curr_sbp < curr_dbp:
            result = SignalQualityResult(
                signal_valid=False,
                artifact_type="bp_inversion",
                artifact_confidence=0.94,
                suppression_recommendation=True,
                artifact_affected_features=["SBP", "DBP", "MAP"],
                suppression_mode="full",
            )
            logs.append(AgentLogEntry(
                agent="Signal Quality",
                message=f"Suppressed: SBP ({curr_sbp:.0f}) < DBP ({curr_dbp:.0f}) — pressure inversion artifact.",
            ))
            return result, logs

        # ── 7. SpO2/HR paradox — window-based HR stability (probe-off heuristic) ──
        if curr_spo2 is not None and curr_spo2 < _SPO2_PARADOX_SPO2 and curr_hr is not None:
            n_check = min(_TREND_WINDOW, len(window_values))
            recent_hrs = [_get(r, "HR", "Pulse") for r in window_values[-n_check:]]
            valid_hrs = [v for v in recent_hrs if v is not None]
            if len(valid_hrs) >= 2:
                hr_stable = (max(valid_hrs) - min(valid_hrs)) < _SPO2_PARADOX_HR_DELTA * 2
            else:
                hr_stable = prev_hr is not None and abs(curr_hr - prev_hr) < _SPO2_PARADOX_HR_DELTA
            if hr_stable:
                result = SignalQualityResult(
                    signal_valid=True,
                    artifact_type="spo2_hr_paradox",
                    artifact_confidence=0.75,
                    suppression_recommendation=True,
                    artifact_affected_features=["O2Sat"],
                    suppression_mode="partial",
                )
                logs.append(AgentLogEntry(
                    agent="Signal Quality",
                    message=(
                        f"Soft suppression: SpO2 {curr_spo2:.0f}% critically low but HR stable "
                        f"over last {n_check} observations — possible probe-off."
                    ),
                ))
                return result, logs

        # ── 8. Flatline detection (multi-feature) ─────────────────────────
        if len(window_values) >= _FLATLINE_MIN_ROWS:
            recent = window_values[-_FLATLINE_MIN_ROWS:]
            flatline_features: list[str] = []
            for feat in ("HR", "Pulse", "SBP", "O2Sat", "Resp"):
                vals = [_get(row, feat) for row in recent]
                valid_vals = [v for v in vals if v is not None]
                if len(valid_vals) >= _FLATLINE_MIN_ROWS and len(set(valid_vals)) == 1:
                    flatline_features.append(feat)

            if flatline_features:
                hard = any(f in _HARD_BLOCK_FEATURES for f in flatline_features)
                mode = "full" if hard else "partial"
                result = SignalQualityResult(
                    signal_valid=not hard,
                    artifact_type="flatline",
                    artifact_confidence=0.85,
                    suppression_recommendation=True,
                    artifact_affected_features=flatline_features,
                    suppression_mode=mode,
                )
                logs.append(AgentLogEntry(
                    agent="Signal Quality",
                    message=f"{'Hard' if hard else 'Soft'} suppression: flatline detected in {', '.join(flatline_features)} over last {_FLATLINE_MIN_ROWS} rows.",
                ))
                return result, logs

        # ── 9. Persistent critically low SpO2 with stable HR ─────────────────
        # In a real ICU, unresolved SpO2 < 85% for 5+ hours with no HR response
        # is almost certainly a probe-off artifact rather than true hypoxia.
        if len(window_values) >= _FLATLINE_MIN_ROWS:
            recent_spo2_rows = window_values[-_FLATLINE_MIN_ROWS:]
            spo2_vals = [_get(r, "O2Sat") for r in recent_spo2_rows]
            valid_spo2 = [v for v in spo2_vals if v is not None]
            if (
                len(valid_spo2) >= _FLATLINE_MIN_ROWS
                and all(v < _PERSISTENT_LOW_SPO2 for v in valid_spo2)
            ):
                hr_vals = [_get(r, "HR", "Pulse") for r in recent_spo2_rows]
                valid_hrs = [v for v in hr_vals if v is not None]
                if len(valid_hrs) >= 2 and (max(valid_hrs) - min(valid_hrs)) < _SPO2_PARADOX_HR_DELTA * 2:
                    result = SignalQualityResult(
                        signal_valid=True,
                        artifact_type="persistent_low_spo2",
                        artifact_confidence=0.82,
                        suppression_recommendation=True,
                        artifact_affected_features=["O2Sat"],
                        suppression_mode="partial",
                    )
                    logs.append(AgentLogEntry(
                        agent="Signal Quality",
                        message=(
                            f"Soft suppression: SpO2 < {_PERSISTENT_LOW_SPO2:.0f}% "
                            f"for {_FLATLINE_MIN_ROWS} consecutive rows with stable HR "
                            f"— probable probe-off artifact."
                        ),
                    ))
                    return result, logs

        # ── Soft range warnings for non-critical vitals ────────────────────
        soft_affected: list[str] = []
        for feat in ("Temp", "Resp", "DBP", "MAP"):
            val = _get(current, feat)
            if val is not None and feat in _BOUNDS:
                lo, hi = _BOUNDS[feat]
                if not (lo <= val <= hi):
                    soft_affected.append(feat)

        if soft_affected:
            result = SignalQualityResult(
                signal_valid=True,
                artifact_type="soft_range_violation",
                artifact_confidence=0.70,
                suppression_recommendation=True,
                artifact_affected_features=soft_affected,
                suppression_mode="partial",
            )
            logs.append(AgentLogEntry(
                agent="Signal Quality",
                message=f"Soft suppression: {', '.join(soft_affected)} outside plausible range.",
            ))
            return result, logs

        # ── All checks passed ──────────────────────────────────────────────
        result = SignalQualityResult(
            signal_valid=True,
            artifact_confidence=0.02,
            suppression_recommendation=False,
            suppression_mode="none",
        )
        logs.append(AgentLogEntry(
            agent="Signal Quality",
            message="Cross-signal sanity checks passed.",
        ))
        return result, logs

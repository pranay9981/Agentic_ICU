"use strict";

// ── Clinical action protocols ─────────────────────────────────────────────
const CLINICAL_ACTIONS = {
  "Sepsis Early Warning": [
    "Blood cultures × 2 before antibiotics",
    "Serum lactate — repeat if > 2 mmol/L",
    "IV broad-spectrum antibiotics within 1 hour",
    "IV fluid resuscitation 30 mL/kg (if hypoperfusion)",
    "Urine output monitoring — insert catheter",
    "Notify intensivist / senior clinician",
  ],
  "Respiratory Failure Risk": [
    "Arterial blood gas (ABG)",
    "Portable chest X-ray",
    "Continuous SpO₂ monitoring",
    "O₂ therapy — target SpO₂ ≥ 94%",
    "NIV assessment if SpO₂ < 90% on ≥ 6 L/min",
    "Respiratory therapy consultation",
  ],
  "Deterioration Watch": [
    "Vital signs every 30 minutes",
    "Senior clinician review within 1 hour",
    "Reassess fluid balance and urine output",
    "Escalation plan if further deterioration",
  ],
  "Suppressed Artifact": [
    "Verify sensor placement and connections",
    "Manual vital sign measurement",
    "Document signal quality concern",
  ],
  "Stable": ["Continue standard monitoring per unit protocol"],
};

function escHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function clinicalLabel(name) {
  const MAP = {
    HR:"Heart Rate", O2Sat:"SpO₂", Resp:"Resp Rate", MAP:"MAP",
    SBP:"Systolic BP", DBP:"Diastolic BP", Temp:"Temperature",
    Lactate:"Lactate", WBC:"WBC", Creatinine:"Creatinine",
    Glucose:"Glucose", Platelets:"Platelets", shock_index:"Shock Index",
    qsofa_score:"qSOFA", pulse_pressure:"Pulse Pressure",
    spo2_fio2_ratio:"SpO₂/FiO₂", map_computed:"MAP (calc)",
  };
  const base = name.replace(/__.*/, "").replace(/_x_.*/, "");
  return MAP[base] || name.replace(/__/g, " ").replace(/_/g, " ");
}

// ── State ─────────────────────────────────────────────────────────────────
const state = {
  patients: new Map(),   // id → { id, window, result, loadedAt, loading }
  selectedId: null,
  timelineData: [],
  timelineRunning: false,
  alertHistory: [],
  alertPolicy: null,
  // compat
  patientId: null, observationWindow: [], monitorIndex: 0,
  lastKnownValues: {}, station: {}, stationPatientCount: 4,
  activeStationPatientIds: [], evaluationRequestId: 0, animationHandle: null,
};

const $ = id => document.getElementById(id);

// ── DOM refs ──────────────────────────────────────────────────────────────
const monitorGrid        = $("monitor-grid");
const detailSidebar      = $("detail-sidebar");
const detailPatientId    = $("detail-patient-id");
const patientLastEval    = $("patient-last-eval");
const decisionCard       = $("decision-card");
const decisionTitle      = $("decision-title");
const decisionRationale  = $("decision-rationale");
const suggestedActions   = $("suggested-actions");
const actionsList        = $("actions-list");
const monitorBed         = $("monitor-bed");
const monitorAlert       = $("monitor-alert");
const monitorSignal      = $("monitor-signal");
const monitorHr          = $("monitor-hr");
const monitorSpo2        = $("monitor-spo2");
const monitorResp        = $("monitor-resp");
const monitorMap         = $("monitor-map");
const monitorTemp        = $("monitor-temp");
const monitorSbp         = $("monitor-sbp");
const monitorSuppression = $("monitor-suppression");
const vitalsBand         = $("vitals-band");
const vitalsScore        = $("vitals-score");
const vitalsDetail       = $("vitals-detail");
const vitalsMeter        = $("vitals-meter");
const vitalsTrace        = $("vitals-trace");
const vitalsThreshold    = $("vitals-threshold");
const vitalsRatio        = $("vitals-ratio");
const labBand            = $("lab-band");
const labScore           = $("lab-score");
const labDetail          = $("lab-detail");
const labMeter           = $("lab-meter");
const labTrace           = $("lab-trace");
const labThreshold       = $("lab-threshold");
const labRatio           = $("lab-ratio");
const respBand           = $("resp-band");
const respScore          = $("resp-score");
const respDetail         = $("resp-detail");
const respMeter          = $("resp-meter");
const respTrace          = $("resp-trace");
const respThreshold      = $("resp-threshold");
const respRatio          = $("resp-ratio");
const labDriversEl       = $("lab-drivers");
const vitalsSaliencyEl   = $("vitals-saliency");
const signalValid        = $("signal-valid");
const artifactType       = $("artifact-type");
const artifactConfidence = $("artifact-confidence");
const suppression        = $("suppression");
const signalCallout      = $("signal-callout");
const reasoningLog       = $("reasoning-log");
const healthGrid         = $("health-grid");
const policySummary      = $("policy-summary");
const reportName         = $("report-name");
const reportSummary      = $("report-summary");
const reportExplainer    = $("report-explainer");
const healthBadge        = $("health-badge");
const alertCounterChip   = $("alert-counter-chip");
const alertCounterEl     = $("alert-counter");
const resultPatient      = $("result-patient");
const alertHistoryLog    = $("alert-history-log");
const clearAlertHistoryBtn = $("clear-alert-history");
const timelineSection    = $("timeline-section");
const riskTimelineSvg    = $("risk-timeline-svg");
const timelineStatus     = $("timeline-status");
const timelineProgress   = $("timeline-progress");
const tpBar              = $("tp-bar");
const tpLabel            = $("tp-label");
const runTimelineBtn     = $("run-timeline");
const patientInput       = $("patient-id");
const countCritical      = $("count-critical");
const countWatch         = $("count-watch");
const countStable        = $("count-stable");
const countTotal         = $("count-total");
const gridStatus         = $("grid-status");

// ── API ───────────────────────────────────────────────────────────────────
const API_TIMEOUT_MS = 30000;

async function apiFetch(path, opts = {}) {
  const controller = new AbortController();
  const tid = setTimeout(() => controller.abort(), API_TIMEOUT_MS);
  try {
    const key = sessionStorage.getItem("icu_api_key") || "";
    const headers = { ...(opts.headers || {}) };
    if (key) headers["X-API-Key"] = key;
    const r = await fetch(path, { ...opts, headers, signal: controller.signal });
    if (r.status === 401) {
      const newKey = prompt("API key required — enter your X-API-Key:");
      if (newKey) { sessionStorage.setItem("icu_api_key", newKey.trim()); return apiFetch(path, opts); }
      throw Object.assign(new Error("Unauthorized"), { status: 401 });
    }
    if (!r.ok) {
      const b = await r.json().catch(() => ({}));
      const msg = Array.isArray(b.detail) ? JSON.stringify(b.detail) : (b.detail || r.statusText);
      throw Object.assign(new Error(msg), { status: r.status });
    }
    return r.json();
  } finally {
    clearTimeout(tid);
  }
}
const apiEvaluate      = (pid, win)   => apiFetch("/evaluate",    {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({patient_id:pid,observation_window:win})});
const apiDemoPatient   = (pid, n=24)  => apiFetch(`/demo-patient/${encodeURIComponent(pid)}?max_rows=${n}`);
const apiDemoPatients  = ()           => apiFetch("/demo-patients");
const apiHealth        = ()           => apiFetch("/health");
const apiConfig        = ()           => apiFetch("/runtime-config");
const apiReport        = ()           => apiFetch("/reports/alert-policy-latest");

// ── SVG sparkline ─────────────────────────────────────────────────────────
function drawSparkline(svgEl, values, {color="#2dd4bf", W=320, H=52, pad=5, zones=[]}={}) {
  if (!svgEl || !values || values.length < 2) { if (svgEl) svgEl.innerHTML = ""; return; }
  const filled = fillGaps(values);
  const fin = filled.filter(v => v != null && isFinite(v));
  if (fin.length < 2) { svgEl.innerHTML = ""; return; }
  const mn = Math.min(...fin), mx = Math.max(...fin), rng = mx - mn || 1;
  const sx = i => pad + (i / (filled.length - 1)) * (W - pad * 2);
  const sy = v => H - pad - ((v - mn) / rng) * (H - pad * 2);
  const pts = filled.map((v, i) => v != null && isFinite(v) ? `${i===0?"M":"L"}${sx(i).toFixed(1)},${sy(v).toFixed(1)}` : null).filter(Boolean).join(" ");
  let z = "";
  for (const zn of zones) {
    const y1 = Math.min(sy(zn.lo), sy(zn.hi)), y2 = Math.max(sy(zn.lo), sy(zn.hi));
    z += `<rect x="0" y="${y1.toFixed(1)}" width="${W}" height="${Math.max(1,(y2-y1)).toFixed(1)}" fill="${zn.color}" opacity="${zn.opacity||0.12}"/>`;
  }
  svgEl.innerHTML = `${z}<path d="${pts}" fill="none" stroke="${color}" stroke-width="1.5" stroke-linejoin="round"/>`;
}

// ── Fill null gaps in a series (forward-fill then backward-fill) ─────────
function fillGaps(values) {
  const out = values.slice();
  // Forward pass: carry last known value forward
  let last = null;
  for (let i = 0; i < out.length; i++) {
    if (out[i] != null && isFinite(out[i])) { last = out[i]; }
    else if (last != null) { out[i] = last; }
  }
  // Backward pass: fill any leading nulls from first valid reading
  last = null;
  for (let i = out.length - 1; i >= 0; i--) {
    if (out[i] != null && isFinite(out[i])) { last = out[i]; }
    else if (last != null) { out[i] = last; }
  }
  return out;
}

// ── Mini sparkline for monitor card ──────────────────────────────────────
function drawMiniSpark(svgEl, values, color, W=120, H=28) {
  if (!svgEl || !values || values.length < 2) return;
  const filled = fillGaps(values);
  const fin = filled.filter(v => v != null && isFinite(v));
  if (fin.length < 2) return;
  const mn = Math.min(...fin), mx = Math.max(...fin), rng = mx - mn || 1;
  const sx = i => 2 + (i / (filled.length - 1)) * (W - 4);
  const sy = v => H - 2 - ((v - mn) / rng) * (H - 4);
  const pts = filled.map((v, i) => v != null && isFinite(v) ? `${i===0?"M":"L"}${sx(i).toFixed(1)},${sy(v).toFixed(1)}` : null).filter(Boolean).join(" ");
  svgEl.innerHTML = `<path d="${pts}" fill="none" stroke="${color}" stroke-width="1.2" stroke-linejoin="round"/>`;
}

function extractSeries(window, key) {
  return window.map(row => { const v = row.values?.[key]; return (v!=null && isFinite(v)) ? v : null; });
}

// ── Live monitoring config + helpers ─────────────────────────────────────
const LIVE = {
  FRAME_MS:  5000,   // ms per hour-frame advance (5s = 1 ICU hour)
  TICK_MS:   900,    // ms for vital-number noise refresh
  NOISE:     { HR:2.5, O2Sat:0.35, Resp:1.2, MAP:3.2 },
  timers:    new Map(),
  tickHandle: null,
};

function noisy(v, noise) {
  if (v == null || !isFinite(v)) return null;
  return v + (Math.random() - 0.5) * 2 * noise;
}

function getDisplayVitals(id) {
  const entry = state.patients.get(id);
  if (!entry?.window?.length) return {};
  const fi  = Math.min(Math.max(0, (entry.currentFrame || entry.window.length) - 1), entry.window.length - 1);
  const v   = entry.window[fi].values || {};
  const map = v.MAP || (v.SBP && v.DBP ? (v.SBP + 2 * v.DBP) / 3 : null);
  return { HR: v.HR, O2Sat: v.O2Sat, Resp: v.Resp, MAP: map, SBP: v.SBP, DBP: v.DBP, Temp: v.Temp, Lactate: v.Lactate };
}

function vitalCls(key, val) {
  if (val == null || !isFinite(val)) return "";
  const ALARMS = { HR:[[0,45],[125,999]], O2Sat:[[0,90]], Resp:[[0,6],[30,99]], MAP:[[0,60]] };
  const WARNS  = { HR:[[45,55],[100,125]], O2Sat:[[90,94]], Resp:[[6,10],[22,30]], MAP:[[60,65]] };
  for (const [lo,hi] of (ALARMS[key]||[])) if (val>=lo && val<hi) return "alarm";
  for (const [lo,hi] of (WARNS[key]||[]))  if (val>=lo && val<hi) return "warn";
  return "";
}

function fmtV(v, dp, key) {
  if (v == null || !isFinite(v)) return { txt:"--", cls:"" };
  return { txt: Number(v).toFixed(dp), cls: vitalCls(key, v) };
}

// ── Render vitals on detail sidebar charts ────────────────────────────────
function renderVitalCharts(window) {
  const CHARTS = [
    {id:"hr-trend-selected",   key:"HR",     color:"#f87171", meta:"trend-hr-selected-meta",   zones:[{lo:50,hi:60,color:"#f87171"},{lo:100,hi:200,color:"#f87171"}]},
    {id:"spo2-trend-selected", key:"O2Sat",  color:"#2dd4bf", meta:"trend-spo2-selected-meta",  zones:[{lo:0,hi:94,color:"#f87171"}]},
    {id:"resp-trend-selected", key:"Resp",   color:"#fbbf24", meta:"trend-resp-selected-meta",  zones:[{lo:20,hi:40,color:"#fbbf24"}]},
    {id:"map-trend",           key:"MAP",    color:"#60a5fa", meta:"trend-map-meta",            zones:[{lo:0,hi:65,color:"#f87171"}]},
    {id:"temp-trend",          key:"Temp",   color:"#a78bfa", meta:"trend-temp-meta",           zones:[{lo:38.3,hi:42,color:"#f87171"}]},
    {id:"lactate-trend",       key:"Lactate",color:"#fb923c", meta:"trend-lactate-meta",        zones:[{lo:2,hi:10,color:"#f87171"}]},
  ];
  for (const c of CHARTS) {
    const svgEl = $(c.id), metaEl = $(c.meta);
    const ser = extractSeries(window, c.key);
    const fin = ser.filter(v => v != null);
    if (fin.length > 0) {
      drawSparkline(svgEl, ser, {color:c.color, zones:c.zones, W:320, H:48});
      if (metaEl) metaEl.textContent = `${fin[fin.length-1].toFixed(1)}  (${Math.min(...fin).toFixed(1)}–${Math.max(...fin).toFixed(1)})`;
    } else {
      if (svgEl) svgEl.innerHTML = "";
      if (metaEl) metaEl.textContent = "No data";
    }
  }
}

function updateVitalTiles(window) {
  if (!window?.length) return;
  const v = window[window.length-1].values || {};
  const f = (val, dp=0) => val!=null && isFinite(val) ? Number(val).toFixed(dp) : "--";
  if (monitorHr)   monitorHr.textContent   = f(v.HR);
  if (monitorSpo2) monitorSpo2.textContent  = f(v.O2Sat, 1);
  if (monitorResp) monitorResp.textContent  = f(v.Resp);
  if (monitorTemp) monitorTemp.textContent  = f(v.Temp, 1);
  if (monitorSbp)  monitorSbp.textContent   = f(v.SBP);
  const map = v.MAP || (v.SBP && v.DBP ? (v.SBP + 2*v.DBP)/3 : null);
  if (monitorMap) monitorMap.textContent = f(map);
}

// ── Score bar helper ──────────────────────────────────────────────────────
function setMeter(el, score) {
  if (el) el.style.width = (Math.min(1, Math.max(0, score||0))*100).toFixed(1)+"%";
}
function setBand(el, band) {
  if (!el) return;
  el.className = `at-band ${band||"muted"}`;
  el.textContent = band==="high" ? "High" : band==="moderate" ? "Watch" : band==="low" ? "Low" : "—";
}

// ── SHAP + saliency ───────────────────────────────────────────────────────
function renderSHAPBars(container, contrib) {
  if (!container) return;
  const items = Object.entries(contrib||{});
  if (!items.length) { container.innerHTML = '<p class="empty-msg">No contributions</p>'; return; }
  const maxAbs = Math.max(...items.map(([,v]) => Math.abs(v)), 0.001);
  container.innerHTML = items.map(([feat, val]) => {
    const pct = (Math.abs(val)/maxAbs*100).toFixed(1);
    return `<div class="shap-row">
      <span class="shap-feat" title="${escHtml(feat)}">${clinicalLabel(feat)}</span>
      <div class="shap-bar-wrap"><div class="shap-bar ${val>=0?"pos":"neg"}" style="width:${pct}%"></div></div>
      <span class="shap-val">${val>=0?"+":""}${val.toFixed(3)}</span>
    </div>`;
  }).join("");
}

function renderSaliencyStrip(container, contrib) {
  if (!container) return;
  const entries = Object.entries(contrib||{}).filter(([k]) => k.startsWith("t_")).sort((a,b)=>a[0].localeCompare(b[0]));
  if (!entries.length) { container.innerHTML = '<p class="empty-msg">No saliency data</p>'; return; }
  const maxW = Math.max(...entries.map(([,v])=>v), 0.001);
  const cells = entries.map(([k,v],i) => {
    const h = Math.round((v/maxW)*34)+2, op = (0.3+(v/maxW)*0.7).toFixed(2);
    return `<div class="sal-cell" title="Hour ${i+1}: ${(v*100).toFixed(1)}%" style="height:${h}px;background:#2dd4bf;opacity:${op}"></div>`;
  }).join("");
  container.innerHTML = `<div class="saliency-strip">${cells}</div>
    <p style="font-size:10px;color:var(--text3);margin-top:3px">Model attention per hour (${entries.length}h window)</p>`;
}

// ── Render agent tiles ────────────────────────────────────────────────────
function renderAgentTiles(result) {
  const {vitals_agent:v, lab_agent:l, resp_failure_agent:r} = result;
  if (v) {
    if (vitalsScore) vitalsScore.textContent = v.score!=null ? v.score.toFixed(3) : "--";
    setMeter(vitalsMeter, v.score); setBand(vitalsBand, v.risk_band);
    if (vitalsThreshold) vitalsThreshold.textContent = v.decision_threshold?.toFixed(3)??"--";
    if (vitalsRatio) vitalsRatio.textContent = v.threshold_ratio != null ? v.threshold_ratio.toFixed(2)+"×" : "--";
    if (vitalsDetail) vitalsDetail.textContent = v.detail||"";
    if (vitalsTrace && v.decision_threshold) setMeter(vitalsTrace, v.decision_threshold);
  }
  if (l) {
    if (labScore) labScore.textContent = l.score!=null ? l.score.toFixed(3) : "--";
    setMeter(labMeter, l.score); setBand(labBand, l.risk_band);
    if (labThreshold) labThreshold.textContent = l.decision_threshold?.toFixed(3)??"--";
    if (labRatio) labRatio.textContent = l.threshold_ratio != null ? l.threshold_ratio.toFixed(2)+"×" : "--";
    if (labDetail) labDetail.textContent = l.detail||"";
    if (labTrace && l.decision_threshold) setMeter(labTrace, l.decision_threshold);
  }
  if (r) {
    if (respScore) respScore.textContent = r.score!=null ? r.score.toFixed(3) : "--";
    setMeter(respMeter, r.score); setBand(respBand, r.risk_band);
    if (respThreshold) respThreshold.textContent = r.decision_threshold?.toFixed(3)??"--";
    if (respRatio) respRatio.textContent = r.threshold_ratio != null ? r.threshold_ratio.toFixed(2)+"×" : "--";
    if (respDetail) respDetail.textContent = r.detail||"";
    if (respTrace && r.decision_threshold) setMeter(respTrace, r.decision_threshold);
  }
}

// ── Render clinical decision ──────────────────────────────────────────────
function renderDecision(result) {
  const cd = result.clinical_decision;
  if (!cd) return;
  const pri = cd.priority||"muted";
  decisionCard.className = `clin-card ${pri}`;
  decisionTitle.textContent = cd.alert_type||"No decision";
  decisionRationale.textContent = cd.rationale||"";
  if (monitorAlert) monitorAlert.textContent = cd.alert_type||"Standby";
  if (monitorBed)   monitorBed.textContent   = result.patient_id||"--";
  const acts = CLINICAL_ACTIONS[cd.alert_type];
  if (acts?.length) {
    if (actionsList) actionsList.innerHTML = acts.map(a=>`<li>${a}</li>`).join("");
    if (suggestedActions) suggestedActions.classList.remove("hidden");
  } else {
    if (suggestedActions) suggestedActions.classList.add("hidden");
  }
}

// ── Render signal quality ─────────────────────────────────────────────────
function renderSignalQuality(result) {
  const sq = result.signal_quality;
  if (!sq) return;
  if (signalValid)        signalValid.textContent        = sq.signal_valid ? "Yes" : "No";
  if (artifactType)       artifactType.textContent       = sq.artifact_type||"None";
  if (artifactConfidence) artifactConfidence.textContent = sq.artifact_confidence!=null ? (sq.artifact_confidence*100).toFixed(0)+"%" : "--";
  if (suppression)        suppression.textContent        = sq.suppression_mode||"none";
  if (monitorSuppression) monitorSuppression.textContent = sq.suppression_mode||"none";
  const affMsg = sq.artifact_affected_features?.length ? `Affected: ${sq.artifact_affected_features.join(", ")}.` : "No features flagged.";
  if (signalCallout) signalCallout.textContent = sq.suppression_mode!=="none"
    ? `Artifact detected — ${sq.suppression_mode} suppression applied. ${affMsg}`
    : "Signal quality validated. No suppression applied.";
  if (monitorSignal) {
    const m = sq.suppression_mode||"none";
    monitorSignal.textContent = m==="none" ? "Signal OK" : m==="partial" ? "Partial Artifact" : "Full Artifact";
    monitorSignal.className = `chip chip-signal ${m==="none"?"ok":"artifact"}`;
  }
  if (reasoningLog && result.reasoning_log) {
    reasoningLog.innerHTML = result.reasoning_log.length
      ? result.reasoning_log.map(e=>`<li><strong>${escHtml(e.agent)}:</strong> ${escHtml(e.message)}</li>`).join("")
      : '<li class="empty-msg">No reasoning trace.</li>';
  }
}

// ── Render detail sidebar ─────────────────────────────────────────────────
function renderSidebar(patientId) {
  const entry = state.patients.get(patientId);
  if (!entry) return;

  if (detailSidebar) detailSidebar.classList.remove("hidden");
  if (detailPatientId) detailPatientId.textContent = patientId;
  if (patientLastEval) patientLastEval.textContent = entry.loadedAt
    ? `Evaluated ${entry.loadedAt.toLocaleTimeString()}` : entry.loading ? "Evaluating…" : "Not evaluated";

  if (!entry.result) return;
  const result = entry.result;
  renderDecision(result);
  renderAgentTiles(result);
  if (labDriversEl) renderSHAPBars(labDriversEl, result.lab_agent?.feature_contributions);
  if (vitalsSaliencyEl) renderSaliencyStrip(vitalsSaliencyEl, result.vitals_agent?.feature_contributions);
  if (entry.window?.length) { updateVitalTiles(entry.window); renderVitalCharts(entry.window); }
  renderSignalQuality(result);
  if (riskTimelineSvg) riskTimelineSvg.innerHTML = "";
  if (timelineSection) timelineSection.classList.add("hidden");
  state.timelineData = [];
}

// ── Monitor card HTML ─────────────────────────────────────────────────────
function buildMonitorCard(entry) {
  const { id, window: win, result, loading, lastError } = entry;
  const cd      = result?.clinical_decision;
  const va      = result?.vitals_agent;
  const ra      = result?.resp_failure_agent;
  const timedOut = lastError === "timeout";
  const pri     = cd?.priority || (loading ? "loading" : "pending");

  const cls      = pri==="high"?"mc-critical":pri==="medium"?"mc-watch":pri==="loading"?"mc-loading":"mc-stable";
  const badge    = pri==="high"?"critical":pri==="medium"?"watch":pri==="loading"?"loading":timedOut?"error":pri==="pending"?"pending":"stable";
  const badgeTxt = pri==="high"?"CRITICAL":pri==="medium"?"WATCH":pri==="loading"?"Evaluating…":timedOut?"TIMEOUT":pri==="pending"?"Pending":"STABLE";
  const alertTxt = cd?.alert_type || (loading ? "Running inference…" : timedOut ? "Request timed out — will retry" : "Awaiting evaluation");

  const cf    = entry.currentFrame || win?.length || 0;
  const total = win?.length || 0;

  // Use currentFrame index for displayed vitals
  const fi  = total ? Math.min(Math.max(0, cf - 1), total - 1) : 0;
  const lv  = total ? (win[fi].values || {}) : {};
  const f   = (v, dp=0) => v!=null && isFinite(v) ? Number(v).toFixed(dp) : "--";
  const map = lv.MAP || (lv.SBP && lv.DBP ? (lv.SBP + 2*lv.DBP)/3 : null);

  const hrC = vitalCls("HR",    lv.HR);
  const spC = vitalCls("O2Sat", lv.O2Sat);
  const rpC = vitalCls("Resp",  lv.Resp);
  const mpC = vitalCls("MAP",   map);

  const sepScore = va?.score!=null ? va.score.toFixed(3) : "--";
  const rspScore = ra?.score!=null ? ra.score.toFixed(3) : "--";
  const sepPct   = va?.score!=null ? (va.score*100).toFixed(1) : "0";
  const rspPct   = ra?.score!=null ? (ra.score*100).toFixed(1) : "0";

  const selected  = id === state.selectedId ? " selected" : "";
  const liveClass = total ? " has-data" : "";
  const footTime  = total ? `H${cf} / H${total}` : "";

  return `<div class="monitor-card ${cls}${selected}${liveClass}" data-patient="${id}" id="mc-${id}">
    <div class="mc-head">
      <span class="mc-id">${id}</span>
      <div class="mc-head-right">
        ${total ? `<span id="mc-frame-${id}" class="mc-frame">H${cf}</span>` : ""}
        <span class="mc-badge ${badge}">${badgeTxt}</span>
        <button class="mc-remove" data-patient="${id}" title="Remove patient" type="button">✕</button>
      </div>
    </div>
    <div class="mc-live-bar">
      ${total ? '<span class="live-dot"></span><span class="live-lbl">LIVE</span>' : '<span class="live-lbl pend">—</span>'}
      <span class="mc-alert-type">${escHtml(alertTxt)}</span>
    </div>
    <div class="mc-mid">
      <div class="mc-vitals">
        <div class="mc-vn hr-v"><span>HR</span>  <strong id="cv-hr-${id}" class="${hrC}">${f(lv.HR)}</strong>   <small>bpm</small></div>
        <div class="mc-vn sp-v"><span>SpO₂</span><strong id="cv-sp-${id}" class="${spC}">${f(lv.O2Sat,1)}</strong><small>%</small></div>
        <div class="mc-vn rp-v"><span>Resp</span><strong id="cv-rp-${id}" class="${rpC}">${f(lv.Resp)}</strong>  <small>bpm</small></div>
        <div class="mc-vn mp-v"><span>MAP</span> <strong id="cv-mp-${id}" class="${mpC}">${f(map)}</strong>      <small>mmHg</small></div>
      </div>
      <div class="mc-sparks">
        <div class="mc-spark-wrap">
          <span id="slbl-hr-${id}" class="mc-spark-lbl">HR ${f(lv.HR)} bpm</span>
          <svg class="mc-spark" id="spark-hr-${id}" viewBox="0 0 120 28" preserveAspectRatio="none"></svg>
        </div>
        <div class="mc-spark-wrap">
          <span id="slbl-sp-${id}" class="mc-spark-lbl">SpO₂ ${f(lv.O2Sat,1)} %</span>
          <svg class="mc-spark" id="spark-sp-${id}" viewBox="0 0 120 28" preserveAspectRatio="none"></svg>
        </div>
      </div>
    </div>
    <div class="mc-scores">
      <div class="mc-score-row">
        <span class="mc-score-lbl">Sepsis</span>
        <div class="mc-score-bar"><div class="mc-score-fill sepsis" style="width:${sepPct}%"></div></div>
        <span class="mc-score-val">${sepScore}</span>
      </div>
      <div class="mc-score-row">
        <span class="mc-score-lbl">Resp</span>
        <div class="mc-score-bar"><div class="mc-score-fill resp" style="width:${rspPct}%"></div></div>
        <span class="mc-score-val">${rspScore}</span>
      </div>
    </div>
    <div class="mc-foot">
      <span id="mc-foot-${id}" class="mc-time">${footTime}</span>
      ${!result && !loading && total ? `<button class="mc-eval-btn" data-patient="${id}" type="button">Evaluate ▶</button>` : ""}
    </div>
  </div>`;
}

// ── Draw sparklines on all monitor cards ──────────────────────────────────
function drawCardSparks() {
  for (const [id, entry] of state.patients) {
    if (!entry.window?.length) continue;
    const cf  = entry.currentFrame || entry.window.length;
    const win = entry.window.slice(0, cf);
    const hrEl = $(`spark-hr-${id}`);
    const spEl = $(`spark-sp-${id}`);
    if (hrEl) drawMiniSpark(hrEl, extractSeries(win, "HR"),    "#f87171", 120, 28);
    if (spEl) drawMiniSpark(spEl, extractSeries(win, "O2Sat"), "#2dd4bf", 120, 28);
  }
}

// ── Update summary chips ──────────────────────────────────────────────────
function updateSummary() {
  let crit=0, watch=0, stable=0, total=0;
  for (const [,e] of state.patients) {
    if (!e.id) continue;
    total++;
    const p = e.result?.clinical_decision?.priority || "low";
    if (p==="high") crit++;
    else if (p==="medium") watch++;
    else stable++;
  }
  if (countCritical) countCritical.textContent = crit;
  if (countWatch)    countWatch.textContent    = watch;
  if (countStable)   countStable.textContent   = stable;
  if (countTotal)    countTotal.textContent    = total;
  if (gridStatus)    gridStatus.textContent    = `${total} patients · ${crit} critical · ${watch} watch`;
}

// ── Live DOM updates (no full re-render needed) ───────────────────────────
function updateCardVitalsDOM(id) {
  const v = getDisplayVitals(id);
  const fields = [
    [`cv-hr-${id}`,  noisy(v.HR,    LIVE.NOISE.HR),    0, "HR"],
    [`cv-sp-${id}`,  noisy(v.O2Sat, LIVE.NOISE.O2Sat), 1, "O2Sat"],
    [`cv-rp-${id}`,  noisy(v.Resp,  LIVE.NOISE.Resp),  0, "Resp"],
    [`cv-mp-${id}`,  noisy(v.MAP,   LIVE.NOISE.MAP),   0, "MAP"],
  ];
  for (const [eid, val, dp, key] of fields) {
    const el = document.getElementById(eid);
    if (!el) continue;
    const { txt, cls } = fmtV(val, dp, key);
    el.textContent = txt;
    el.className = cls;
  }
  const hrLbl = document.getElementById(`slbl-hr-${id}`);
  const spLbl = document.getElementById(`slbl-sp-${id}`);
  if (hrLbl) hrLbl.textContent = `HR ${fmtV(noisy(v.HR, LIVE.NOISE.HR), 0, "HR").txt} bpm`;
  if (spLbl) spLbl.textContent = `SpO₂ ${fmtV(noisy(v.O2Sat, LIVE.NOISE.O2Sat), 1, "O2Sat").txt} %`;
}

function updateSidebarVitalsDOM(id) {
  if (id !== state.selectedId) return;
  const v = getDisplayVitals(id);
  const f = (val, dp=0) => val != null && isFinite(val) ? Number(val).toFixed(dp) : "--";
  if (monitorHr)   monitorHr.textContent   = f(noisy(v.HR,    LIVE.NOISE.HR),    0);
  if (monitorSpo2) monitorSpo2.textContent  = f(noisy(v.O2Sat, LIVE.NOISE.O2Sat), 1);
  if (monitorResp) monitorResp.textContent  = f(noisy(v.Resp,  LIVE.NOISE.Resp),  0);
  if (monitorMap)  monitorMap.textContent   = f(noisy(v.MAP,   LIVE.NOISE.MAP),   0);
  if (monitorTemp) monitorTemp.textContent  = f(v.Temp, 1);
  if (monitorSbp)  monitorSbp.textContent   = f(v.SBP, 0);
}

function updateCardScore(id) {
  const card  = document.getElementById(`mc-${id}`);
  const entry = state.patients.get(id);
  if (!card || !entry?.result) return;
  const cd  = entry.result.clinical_decision;
  const va  = entry.result.vitals_agent;
  const ra  = entry.result.resp_failure_agent;
  const pri = cd?.priority || "low";
  const sel = id === state.selectedId ? " selected" : "";
  card.className = `monitor-card ${pri==="high"?"mc-critical":pri==="medium"?"mc-watch":"mc-stable"}${sel} has-data`;
  const badge = card.querySelector(".mc-badge");
  if (badge) {
    badge.className = `mc-badge ${pri==="high"?"critical":pri==="medium"?"watch":"stable"}`;
    badge.textContent = pri==="high"?"CRITICAL":pri==="medium"?"WATCH":"STABLE";
  }
  const alertEl = card.querySelector(".mc-alert-type");
  if (alertEl) alertEl.textContent = cd?.alert_type || "Stable";
  const s1 = card.querySelector(".mc-score-fill.sepsis");
  const s2 = card.querySelector(".mc-score-fill.resp");
  const v1 = card.querySelector(".mc-score-row:first-child .mc-score-val");
  const v2 = card.querySelector(".mc-score-row:last-child  .mc-score-val");
  if (s1 && va?.score!=null) s1.style.width = (va.score*100).toFixed(1)+"%";
  if (s2 && ra?.score!=null) s2.style.width = (ra.score*100).toFixed(1)+"%";
  if (v1 && va?.score!=null) v1.textContent = va.score.toFixed(3);
  if (v2 && ra?.score!=null) v2.textContent = ra.score.toFixed(3);
  updateSummary();
}

// ── Render full monitor grid ──────────────────────────────────────────────
function renderGrid() {
  if (!monitorGrid) return;

  const patients = [...state.patients.values()].filter(e => e.id);
  // Sort: critical first, then watch, then stable, then loading
  const order = p => p==="high"?0 : p==="medium"?1 : p==="loading"?3 : 2;
  patients.sort((a,b) => {
    const pa = a.result?.clinical_decision?.priority || (a.loading?"loading":"low");
    const pb = b.result?.clinical_decision?.priority || (b.loading?"loading":"low");
    return order(pa) - order(pb) || a.id.localeCompare(b.id);
  });

  if (patients.length === 0) {
    monitorGrid.innerHTML = '<div class="grid-loading"><span>No patients loaded</span></div>';
    return;
  }

  monitorGrid.innerHTML = patients.map(e => buildMonitorCard(e)).join("");
  drawCardSparks();
  updateSummary();

  // Attach click listeners
  monitorGrid.querySelectorAll(".monitor-card").forEach(card => {
    card.addEventListener("click", e => {
      if (e.target.classList.contains("mc-eval-btn") || e.target.classList.contains("mc-remove")) return;
      selectPatient(card.dataset.patient);
    });
  });
  monitorGrid.querySelectorAll(".mc-eval-btn").forEach(btn => {
    btn.addEventListener("click", e => { e.stopPropagation(); evaluatePatient(btn.dataset.patient); });
  });
  monitorGrid.querySelectorAll(".mc-remove").forEach(btn => {
    btn.addEventListener("click", e => { e.stopPropagation(); removePatient(btn.dataset.patient); });
  });
}

// ── Remove patient ────────────────────────────────────────────────────────
function removePatient(id) {
  if (LIVE.timers.has(id)) { clearTimeout(LIVE.timers.get(id)); LIVE.timers.delete(id); }
  if (state.selectedId === id) {
    state.selectedId = null;
    if (detailSidebar) detailSidebar.classList.add("hidden");
  }
  state.patients.delete(id);
  saveBoardState();
  renderGrid();
}

// ── Patient selection modal ───────────────────────────────────────────────
let pmSearch      = "";
let pmTotal       = 0;
let pmResults     = [];
let pmSearchTimer = null;
const PM_LIMIT    = 100;

// ── Modal focus trap ──────────────────────────────────────────────────────
function _trapFocus(modal, e) {
  if (e.key !== "Tab") return;
  const sel = 'a[href],button:not([disabled]),input:not([disabled]),select:not([disabled]),textarea:not([disabled]),[tabindex]:not([tabindex="-1"])';
  const focusable = [...modal.querySelectorAll(sel)].filter(el => el.offsetParent !== null);
  if (focusable.length < 2) return;
  const first = focusable[0], last = focusable[focusable.length - 1];
  if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
  else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
}

function _attachTrap(modal) {
  if (!modal) return;
  if (modal._focusTrap) modal.removeEventListener("keydown", modal._focusTrap);
  modal._focusTrap = e => _trapFocus(modal, e);
  modal.addEventListener("keydown", modal._focusTrap);
}

function _detachTrap(modal) {
  if (!modal?._focusTrap) return;
  modal.removeEventListener("keydown", modal._focusTrap);
  delete modal._focusTrap;
}

function closePatientModal() {
  const backdrop = $("patient-modal-backdrop");
  if (!backdrop) return;
  _detachTrap(backdrop.querySelector('[role="dialog"]'));
  backdrop.classList.add("hidden");
  backdrop.setAttribute("aria-hidden", "true");
  $("browse-patients-btn")?.focus();
}

function renderPatientRows(items) {
  const list = $("pm-list");
  if (!list) return;
  if (!items.length) {
    list.innerHTML = '<div class="pm-no-results">No patients found — try a different search</div>';
    return;
  }

  const rows = items.map(p => {
    const onBoard = state.patients.has(p.id);
    const dot   = p.tone ? `<span class="pm-risk-dot ${escHtml(p.tone)}"></span>` : `<span class="pm-risk-dot low"></span>`;
    const label = p.label ? `<span class="pm-label ${escHtml(p.tone||"low")}">${escHtml(p.label)}</span>` : "";
    return `<div class="pm-row${onBoard?" on-board":""}" data-id="${escHtml(p.id)}">
      ${dot}
      <span class="pm-pid">${escHtml(p.id)}</span>
      ${label}
      <span class="pm-spacer"></span>
      ${onBoard
        ? `<span class="pm-on-board-badge">● On board</span>`
        : `<button class="pm-add-btn" data-id="${p.id}" type="button">Add to Board</button>`}
    </div>`;
  }).join("");

  list.innerHTML = rows;

  // Keyboard navigation: ArrowDown from search focuses first row button;
  // ArrowUp/ArrowDown navigates between row buttons.
  const searchInput = $("pm-search");
  if (searchInput) {
    searchInput.onkeydown = e => {
      if (e.key === "ArrowDown") {
        e.preventDefault();
        list.querySelector(".pm-add-btn, .pm-on-board-badge")?.focus();
      }
    };
  }
  list.addEventListener("keydown", e => {
    const btns = [...list.querySelectorAll(".pm-add-btn")];
    const idx  = btns.indexOf(document.activeElement);
    if (e.key === "ArrowDown" && idx < btns.length - 1) { e.preventDefault(); btns[idx+1].focus(); }
    if (e.key === "ArrowUp"   && idx > 0)               { e.preventDefault(); btns[idx-1].focus(); }
    if (e.key === "ArrowUp"   && idx === 0)              { e.preventDefault(); searchInput?.focus(); }
  });
  // Wire add buttons
  list.querySelectorAll(".pm-add-btn").forEach(btn => {
    btn.addEventListener("click", e => {
      e.stopPropagation();
      const id = btn.dataset.id;
      btn.textContent = "Adding…";
      btn.disabled = true;
      loadPatient(id, true);
      setTimeout(() => {
        const row = list.querySelector(`[data-id="${id}"]`);
        if (row) {
          row.classList.add("on-board");
          row.querySelector(".pm-add-btn")?.replaceWith(
            Object.assign(document.createElement("span"), { className:"pm-on-board-badge", textContent:"● On board" })
          );
        }
      }, 500);
    });
  });
}

async function fetchPatients() {
  const list = $("pm-list");
  const countEl = $("pm-count");
  if (!list) return;

  list.innerHTML = '<div class="pm-loading"><div class="spinner-lg"></div><span>Searching…</span></div>';

  try {
    const params = new URLSearchParams({ limit: PM_LIMIT });
    if (pmSearch) params.set("search", pmSearch);
    const data = await apiFetch(`/patients?${params}`);
    pmResults = data.patients;
    pmTotal   = data.total;
    if (countEl) countEl.textContent = pmSearch
      ? `Top ${pmResults.length} of ${pmTotal.toLocaleString()} matches`
      : `Showing ${pmResults.length} of ${pmTotal.toLocaleString()} — search to filter`;
    const titleEl = $("pm-title");
    if (titleEl && !pmSearch) titleEl.textContent = `Browse Patients`;
    renderPatientRows(pmResults);
  } catch(err) {
    console.error("[fetchPatients]:", err);
    list.innerHTML = '<div class="pm-no-results">Failed to load patients</div>';
  }
}

function pmDebounceSearch(val) {
  pmSearch = val;
  clearTimeout(pmSearchTimer);
  pmSearchTimer = setTimeout(() => fetchPatients(), 280);
}

async function openPatientModal() {
  const backdrop = $("patient-modal-backdrop");
  if (!backdrop) return;
  backdrop.classList.remove("hidden");
  backdrop.removeAttribute("aria-hidden");
  pmSearch = "";
  const searchEl = $("pm-search");
  if (searchEl) { searchEl.value = ""; searchEl.focus(); }
  document.querySelectorAll(".pm-filter").forEach(b => b.classList.toggle("active", b.dataset.filter === "all"));
  _attachTrap(backdrop.querySelector('[role="dialog"]'));
  fetchPatients();
}

// ── Timeline analysis ─────────────────────────────────────────────────────
async function runTimelineAnalysis(patientId) {
  const entry = state.patients.get(patientId);
  if (!entry?.window?.length || state.timelineRunning) return;
  state.timelineRunning = true;
  const win = entry.window, total = win.length;
  state.timelineData = [];
  if (timelineSection)  timelineSection.classList.remove("hidden");
  if (timelineProgress) timelineProgress.classList.remove("hidden");
  if (runTimelineBtn)   runTimelineBtn.classList.add("running");
  if (timelineStatus)   timelineStatus.textContent = "Evaluating…";

  const CONCURRENCY = 4;
  let done = 0;
  while (done < total && state.selectedId === patientId) {
    const batchEnd = Math.min(done + CONCURRENCY, total);
    const frames = Array.from({length: batchEnd - done}, (_, i) => done + i + 1);
    const items = await Promise.all(frames.map(f =>
      apiEvaluate(patientId, win.slice(0, f))
        .then(r => ({hour:f, sepsis:r.vitals_agent?.score??0, resp:r.resp_failure_agent?.score??0}))
        .catch(err => { console.error(`[timeline] frame ${f}:`, err); return null; })
    ));
    for (const item of items) if (item) state.timelineData.push(item);
    state.timelineData.sort((a,b) => a.hour - b.hour);
    drawTimelineSvg(state.timelineData, total);
    done = batchEnd;
    if (tpBar)   tpBar.style.width = (done/total*100).toFixed(0)+"%";
    if (tpLabel) tpLabel.textContent = `Frame ${done} / ${total}`;
  }
  state.timelineRunning = false;
  if (runTimelineBtn)   runTimelineBtn.classList.remove("running");
  if (timelineProgress) timelineProgress.classList.add("hidden");
  if (timelineStatus) {
    const last = state.timelineData[state.timelineData.length-1];
    timelineStatus.textContent = last ? `Sepsis ${last.sepsis.toFixed(3)} · Resp ${last.resp.toFixed(3)} at ${total}h` : "Done";
  }
}

function drawTimelineSvg(results, maxH) {
  if (!riskTimelineSvg || !results?.length) return;
  const W=800, H=160, px=36, py=10;
  const gW=W-px*2, gH=H-py*2;
  const sx = h => px + ((h-1)/Math.max(maxH-1,1))*gW;
  const sy = v => py + (1-v)*gH;
  const ap = state.alertPolicy || {};
  const th_extreme = ap.high_alert_extreme_sequence_score_threshold ?? 0.88;
  const th_watch   = ap.high_alert_supported_sequence_score_threshold ?? 0.80;
  const th_medium  = ap.medium_alert_sequence_score_threshold ?? 0.55;
  const zh=sy(th_extreme), zw=sy(th_watch), zm=sy(th_medium);
  let s = `<rect x="${px}" y="${py}"  width="${gW}" height="${zh-py}"  fill="rgba(248,113,113,.07)"/>
    <rect x="${px}" y="${zh}" width="${gW}" height="${zw-zh}" fill="rgba(251,191,36,.05)"/>
    <line x1="${px}" y1="${zh}" x2="${W-px}" y2="${zh}" stroke="rgba(248,113,113,.5)" stroke-width=".7" stroke-dasharray="4,3"/>
    <line x1="${px}" y1="${zw}" x2="${W-px}" y2="${zw}" stroke="rgba(251,191,36,.5)"  stroke-width=".7" stroke-dasharray="4,3"/>
    <line x1="${px}" y1="${zm}" x2="${W-px}" y2="${zm}" stroke="rgba(251,191,36,.35)" stroke-width=".7" stroke-dasharray="4,3"/>`;
  for (const h of [1,4,8,12,16,20,maxH]) {
    if (h>maxH) continue;
    const x=sx(h);
    s += `<line x1="${x}" y1="${py}" x2="${x}" y2="${H-py}" stroke="rgba(255,255,255,.05)" stroke-width=".7"/>
      <text x="${x}" y="${H-1}" text-anchor="middle" fill="#415e7a" font-size="9">${h}h</text>`;
  }
  const spts = results.map((r,i)=>`${i===0?"M":"L"}${sx(r.hour).toFixed(1)},${sy(r.sepsis).toFixed(1)}`).join(" ");
  const rpts = results.map((r,i)=>`${i===0?"M":"L"}${sx(r.hour).toFixed(1)},${sy(r.resp).toFixed(1)}`).join(" ");
  s += `<path d="${spts}" fill="none" stroke="#2dd4bf" stroke-width="2" stroke-linejoin="round"/>
    <path d="${rpts}" fill="none" stroke="#a78bfa" stroke-width="1.5" stroke-linejoin="round" stroke-dasharray="4,2"/>`;
  for (const r of results) {
    s += `<circle cx="${sx(r.hour).toFixed(1)}" cy="${sy(r.sepsis).toFixed(1)}" r="2.5" fill="#2dd4bf"/>`;
  }
  riskTimelineSvg.innerHTML = s;
}

// ── Select patient ────────────────────────────────────────────────────────
async function selectPatient(patientId) {
  state.selectedId = patientId;
  state.patientId  = patientId;
  renderGrid();        // re-render to show selected state
  renderSidebar(patientId);
  const entry = state.patients.get(patientId);
  if (entry && !entry.result && !entry.loading) {
    await evaluatePatient(patientId);
  }
}

// ── Load patient window ───────────────────────────────────────────────────
async function loadPatientData(patientId) {
  state.patients.set(patientId, {id:patientId, window:null, result:null, loadedAt:null, loading:true, currentFrame:0});
  renderGrid();
  try {
    const demo = await apiDemoPatient(patientId, 24);
    const win  = demo.observation_window;
    state.patients.set(patientId, {id:patientId, window:win, result:null, loadedAt:null, loading:false, currentFrame:win.length});
    renderGrid();
  } catch(err) {
    state.patients.set(patientId, {id:patientId, window:null, result:null, loadedAt:null, loading:false, currentFrame:0, error:err.message});
    renderGrid();
  }
}

// ── Evaluate patient ──────────────────────────────────────────────────────
async function evaluatePatient(patientId) {
  const entry = state.patients.get(patientId);
  if (!entry?.window) return;
  state.patients.set(patientId, {...entry, loading:true, lastError: null});
  renderGrid();
  if (state.selectedId === patientId) renderSidebar(patientId);
  try {
    const result = await apiEvaluate(patientId, entry.window);
    state.patients.set(patientId, {...state.patients.get(patientId), result, loadedAt:new Date(), loading:false, lastError: null});
    recordAlert(patientId, result);
    renderGrid();
    if (state.selectedId === patientId) renderSidebar(patientId);
  } catch(err) {
    const isTimeout = err.name === "AbortError";
    console.error(`[evaluatePatient] ${patientId}:`, isTimeout ? "Request timed out (30s)" : err);
    state.patients.set(patientId, {...state.patients.get(patientId), loading:false, lastError: isTimeout ? "timeout" : null});
    renderGrid();
  }
}

// ── Load + evaluate ───────────────────────────────────────────────────────
async function loadPatient(patientId, autoSelect=false) {
  await loadPatientData(patientId);
  if (autoSelect) { state.selectedId = patientId; renderSidebar(patientId); }
  await evaluatePatient(patientId);
  if (autoSelect) renderSidebar(patientId);
  saveBoardState();
  startLiveForPatient(patientId);
}

// ── Re-evaluate selected ──────────────────────────────────────────────────
async function reEvaluate() {
  const id = state.selectedId;
  if (!id) return;
  const entry = state.patients.get(id);
  const btn = $("toggle-playback");
  if (btn) { btn.classList.add("loading"); btn.textContent = "Evaluating…"; }
  try {
    const currentRows = entry?.window?.length || 24;
    const demo = await apiDemoPatient(id, currentRows);
    const result = await apiEvaluate(id, demo.observation_window);
    state.patients.set(id, {...state.patients.get(id), window:demo.observation_window, result, loadedAt:new Date(), loading:false});
    recordAlert(id, result);
    renderGrid();
    renderSidebar(id);
  } catch(err) {
    console.error("[reEvaluate]:", err);
  }
  if (btn) { btn.classList.remove("loading"); btn.textContent = "Re-evaluate"; }
}

// ── Alert history ─────────────────────────────────────────────────────────
function recordAlert(pid, result) {
  const cd = result?.clinical_decision;
  if (!cd?.alert_triggered) return;
  state.alertHistory.unshift({time:new Date().toLocaleTimeString(), patientId:pid, alertType:cd.alert_type, priority:cd.priority||"low"});
  if (state.alertHistory.length > 500) state.alertHistory.length = 500;
  try { sessionStorage.setItem("icu_alerts", JSON.stringify(state.alertHistory)); } catch(_) {}
  renderAlertHistory();
  updateAlertCounter();
  const announcer = $("alert-announcer");
  if (announcer) announcer.textContent = `New alert: ${cd.alert_type} for patient ${pid}`;
}
function loadAlertHistory() {
  try { const s=sessionStorage.getItem("icu_alerts"); if (s) state.alertHistory=JSON.parse(s); } catch(_) {}
  renderAlertHistory(); updateAlertCounter();
}
function renderAlertHistory() {
  if (!alertHistoryLog) return;
  alertHistoryLog.innerHTML = state.alertHistory.length
    ? state.alertHistory.map(a=>`<li class="ah-item ${escHtml(a.priority)}"><span class="ah-time">${escHtml(a.time)}</span><span class="ah-patient">${escHtml(a.patientId)}</span><span class="ah-type">${escHtml(a.alertType)}</span></li>`).join("")
    : '<li class="empty-msg">No alerts yet.</li>';
}
function updateAlertCounter() {
  const n = state.alertHistory.length;
  if (alertCounterChip) alertCounterChip.classList.toggle("hidden", n===0);
  if (alertCounterEl)   alertCounterEl.textContent = n;
}

// ── Health + config ───────────────────────────────────────────────────────
async function refreshHealth() {
  try {
    const h = await apiHealth();
    const ok = h.xgboost_ready && h.sequence_ready;
    if (healthBadge) { healthBadge.className=`health-pill ${ok?"ok":"degraded"}`; healthBadge.textContent=ok?"ONLINE":"DEGRADED"; }
    if (healthGrid) healthGrid.innerHTML = [
      {l:"Preprocessing",ok:h.preprocessing_ready},
      {l:"XGBoost (Lab)", ok:h.xgboost_ready},
      {l:"GRU (Vitals)",  ok:h.sequence_ready},
      {l:"GRU (Resp)",    ok:h.resp_ready},
      {l:`${h.patient_count??0} patients`,ok:true},
      {l:`Load ${h.load_latency_ms??0}ms`,ok:true},
    ].map(r=>`<div class="health-row"><span>${r.l}</span><span class="${r.ok?"ok":"fail"}">${r.ok?"●":"✗"}</span></div>`).join("");
  } catch(_) {
    if (healthBadge) { healthBadge.className="health-pill error"; healthBadge.textContent="OFFLINE"; }
  }
  try {
    const cfg = await apiConfig();
    if (cfg.alert_policy) state.alertPolicy = cfg.alert_policy;
    if (policySummary && cfg.alert_policy) {
      const p = cfg.alert_policy;
      policySummary.innerHTML = [
        ["Extreme threshold",  p.high_alert_extreme_sequence_score_threshold],
        ["High supported",     p.high_alert_supported_sequence_score_threshold],
        ["Medium sequence",    p.medium_alert_sequence_score_threshold],
        ["Medium tabular",     p.medium_alert_tabular_score_threshold],
        ["Resp high",          p.resp_high_alert_threshold],
        ["Resp medium",        p.resp_medium_alert_threshold],
      ].map(([k,v])=>`<div class="policy-row"><span>${k}</span><span>${v??"-"}</span></div>`).join("");
    }
  } catch(_) {}
  try {
    const rep = await apiReport();
    if (reportName) reportName.textContent = rep.report_name||"—";
    if (reportSummary) reportSummary.innerHTML = `<div class="policy-row"><span>Patients evaluated</span><span>${rep.patients_evaluated??"-"}</span></div>`;
    if (reportExplainer && rep.best_profile_by_balanced_accuracy) reportExplainer.textContent = `Best profile: ${rep.best_profile_by_balanced_accuracy}`;
  } catch(_) {
    if (reportName) reportName.textContent = "No calibration report saved";
    if (reportExplainer) reportExplainer.innerHTML = '<p class="empty-msg">Recommendation notes will appear here.</p>';
  }
}

// ── Clock ─────────────────────────────────────────────────────────────────
function startClock() {
  const el = $("clock");
  if (!el) return;
  const tick = () => { el.textContent = new Date().toLocaleTimeString(); };
  tick(); setInterval(tick, 1000);
}

// ── Grid columns ──────────────────────────────────────────────────────────
function applyGridCols(n) {
  const grid = monitorGrid;
  if (grid) grid.style.setProperty("--grid-cols", n);
  // Also set on .monitor-grid via inline style for older browsers
  if (grid) grid.style.gridTemplateColumns = `repeat(${n},1fr)`;
}

// ── Live monitoring engine ────────────────────────────────────────────────
function liveTick() {
  for (const [id, entry] of state.patients) {
    if (!entry.window?.length) continue;
    updateCardVitalsDOM(id);
    updateSidebarVitalsDOM(id);
  }
}

async function advanceLiveFrame(id) {
  const entry = state.patients.get(id);
  if (!entry?.window?.length) {
    LIVE.timers.set(id, setTimeout(() => advanceLiveFrame(id), LIVE.FRAME_MS));
    return;
  }
  const total   = entry.window.length;
  const current = entry.currentFrame || total;
  const next    = current >= total ? 1 : current + 1;

  state.patients.set(id, { ...state.patients.get(id), currentFrame: next });

  // Update sparkline up to current frame (no full re-render)
  const frameWin = entry.window.slice(0, next);
  const hrEl = document.getElementById(`spark-hr-${id}`);
  const spEl = document.getElementById(`spark-sp-${id}`);
  if (hrEl) drawMiniSpark(hrEl, extractSeries(frameWin, "HR"),    "#f87171", 120, 28);
  if (spEl) drawMiniSpark(spEl, extractSeries(frameWin, "O2Sat"), "#2dd4bf", 120, 28);

  // Frame counter + footer
  const fc = document.getElementById(`mc-frame-${id}`);
  if (fc) fc.textContent = `H${next}`;
  const ft = document.getElementById(`mc-foot-${id}`);
  if (ft) ft.textContent = `H${next} / H${total}`;

  // Re-evaluate every 3 frames or when looping back to H1 — skip when tab is hidden
  if ((next % 3 === 0 || next === 1) && !document.hidden) {
    evaluatePatientSilent(id, frameWin);
  }

  LIVE.timers.set(id, setTimeout(() => advanceLiveFrame(id), LIVE.FRAME_MS));
}

async function evaluatePatientSilent(id, win) {
  const entry = state.patients.get(id);
  if (!entry) return;
  try {
    const result = await apiEvaluate(id, win || entry.window);
    state.patients.set(id, { ...state.patients.get(id), result, loadedAt: new Date() });
    recordAlert(id, result);
    updateCardScore(id);
    if (state.selectedId === id) renderSidebar(id);
  } catch (err) {
    if (err.name !== "AbortError") console.error(`[evaluatePatientSilent] ${id}:`, err);
  }
}

function startLiveMonitoring() {
  // Start vital number tick (noise every ~1s)
  if (LIVE.tickHandle) clearInterval(LIVE.tickHandle);
  LIVE.tickHandle = setInterval(liveTick, LIVE.TICK_MS);

  // Stagger frame advances across patients to avoid simultaneous evaluate calls
  let delay = 0;
  for (const [id, entry] of state.patients) {
    if (!entry.window?.length) continue;
    state.patients.set(id, { ...entry, currentFrame: entry.window.length });
    setTimeout(() => advanceLiveFrame(id), delay);
    delay += 1800; // 1.8s stagger so evaluations don't pile up
  }
}

function startLiveForPatient(id) {
  const entry = state.patients.get(id);
  if (!entry?.window?.length) return;
  if (LIVE.timers.has(id)) clearTimeout(LIVE.timers.get(id));
  state.patients.set(id, { ...state.patients.get(id), currentFrame: entry.window.length });
  setTimeout(() => advanceLiveFrame(id), 1200);
}

// ── Board persistence ─────────────────────────────────────────────────────
const BOARD_KEY   = "icu_board_patients";
const DEFAULT_IDS = ["p000001","p000026","p000028","p000002","p000004","p000005","p000006","p000011"];

function saveBoardState() {
  try {
    const ids = [...state.patients.keys()];
    localStorage.setItem(BOARD_KEY, JSON.stringify(ids));
  } catch(_) {}
}

function loadBoardIds() {
  try {
    const saved = localStorage.getItem(BOARD_KEY);
    if (saved) {
      const ids = JSON.parse(saved);
      if (Array.isArray(ids) && ids.length > 0) return ids;
    }
  } catch(_) {}
  return DEFAULT_IDS;
}

// ── Model Metrics Modal ───────────────────────────────────────────────────
const MM_COLORS = {
  teal:"#2dd4bf", amber:"#fbbf24", violet:"#a78bfa", blue:"#60a5fa",
  text2:"#7a9bbf", text3:"#415e7a", border:"#1a2d48",
};
const MM_MODEL_COLORS = {sepsis_gru:"teal", sepsis_xgb:"amber", resp_gru:"violet", resp_xgb:"blue"};

let _metricsData = null;
let _metricsTab  = "overview";

function openMetricsModal() {
  const bd = $("metrics-modal-backdrop");
  if (!bd) return;
  bd.classList.remove("hidden");
  bd.removeAttribute("aria-hidden");
  _attachTrap(bd.querySelector('[role="dialog"]'));
  bd.querySelector(".pm-close-btn")?.focus();
  if (!_metricsData) fetchModelMetrics();
  else renderMetricsTab(_metricsTab);
}

function closeMetricsModal() {
  const bd = $("metrics-modal-backdrop");
  if (!bd) return;
  _detachTrap(bd.querySelector('[role="dialog"]'));
  bd.classList.add("hidden");
  bd.setAttribute("aria-hidden", "true");
  $("model-metrics-btn")?.focus();
}

async function fetchModelMetrics() {
  const body = $("mm-body");
  if (!body) return;
  body.innerHTML = '<div class="pm-loading"><div class="spinner-lg"></div><span>Loading model metrics…</span></div>';
  try {
    _metricsData = await apiFetch("/model-metrics");
    renderMetricsTab(_metricsTab);
  } catch(e) {
    body.innerHTML = `<div class="pm-loading"><span style="color:var(--red)">Failed to load: ${escHtml(e.message)}</span><button class="gt-btn mm-retry-btn" style="margin-top:12px;padding:6px 18px">Retry</button></div>`;
    body.querySelector(".mm-retry-btn")?.addEventListener("click", fetchModelMetrics);
  }
}

function renderMetricsTab(tab) {
  _metricsTab = tab;
  document.querySelectorAll(".mm-tab").forEach(b => {
    const isActive = b.dataset.tab === tab;
    b.classList.toggle("active", isActive);
    b.setAttribute("aria-selected", isActive ? "true" : "false");
  });
  const body = $("mm-body");
  if (!body || !_metricsData) return;
  if (tab === "overview") {
    body.innerHTML = buildMMOverview();
    requestAnimationFrame(drawMMComparisonChart);
  } else {
    const m = _metricsData[tab];
    body.innerHTML = buildMMDetail(m, tab);
    requestAnimationFrame(() => {
      if (m.type === "sequence") drawMMTrainingCurve(m.history);
      else drawMMFeatureChart(m.top_features);
    });
  }
}

// ── Overview ──────────────────────────────────────────────────────────────
function buildMMOverview() {
  const d = _metricsData;
  const cards = ["sepsis_gru","sepsis_xgb","resp_gru","resp_xgb"].map(key => {
    const m   = d[key];
    const met = m.metrics;
    const c   = MM_MODEL_COLORS[key];
    return `
      <div class="mm-model-card mm-card-${c}">
        <div class="mm-card-head">
          <span class="mm-model-name">${m.name}</span>
          <span class="mm-model-badge mm-badge-${c}">${m.type}</span>
        </div>
        <div class="mm-kpi-row">
          <div class="mm-kpi"><div class="mm-kpi-label">AUC-ROC</div><div class="mm-kpi-val mm-c-${c}">${met.auc.toFixed(3)}</div></div>
          <div class="mm-kpi"><div class="mm-kpi-label">AUPRC</div><div class="mm-kpi-val mm-c-${c}">${met.average_precision.toFixed(3)}</div></div>
          <div class="mm-kpi"><div class="mm-kpi-label">F1</div><div class="mm-kpi-val">${met.f1.toFixed(3)}</div></div>
        </div>
        <div class="mm-gauge-list">
          ${mmGauge("Precision", met.precision, c)}
          ${mmGauge("Recall",    met.recall,    c)}
          ${mmGauge("F1-Score",  met.f1,        c)}
        </div>
        <div class="mm-card-foot">
          Brier <strong>${met.brier_score.toFixed(4)}</strong>
          &nbsp;·&nbsp; Threshold <strong>${met.threshold.toFixed(3)}</strong>
        </div>
      </div>`;
  }).join("");

  return `
    <div class="mm-overview-grid">${cards}</div>
    <div class="mm-chart-section">
      <div class="mm-chart-title">AUC · AUPRC · F1 — Side-by-Side Comparison</div>
      <svg id="mm-comparison-chart" class="mm-svg" viewBox="0 0 760 190" preserveAspectRatio="xMidYMid meet"></svg>
      <div class="mm-chart-legend">
        <span class="mm-leg" style="background:rgba(255,255,255,.35)">■ AUC</span>
        <span class="mm-leg" style="background:rgba(255,255,255,.55)">■ AUPRC</span>
        <span class="mm-leg" style="background:rgba(255,255,255,.75)">■ F1</span>
      </div>
    </div>`;
}

function mmGauge(label, val, colorKey) {
  const pct = Math.min(100, Math.round(val * 100));
  return `<div class="mm-gauge">
    <span class="mm-gauge-lbl">${label}</span>
    <div class="mm-gauge-track"><div class="mm-gauge-fill mm-bg-${colorKey}" style="width:${pct}%"></div></div>
    <span class="mm-gauge-pct">${(val*100).toFixed(1)}%</span>
  </div>`;
}

// ── Detail ────────────────────────────────────────────────────────────────
function buildMMDetail(m, key) {
  const met = m.metrics;
  const c   = MM_MODEL_COLORS[key];
  const metricRows = [
    ["AUC-ROC",            met.auc.toFixed(4),                    ""],
    ["Avg Precision (AP)", met.average_precision.toFixed(4),       ""],
    ["Precision",          `${(met.precision*100).toFixed(1)}%`,   ""],
    ["Recall",             `${(met.recall*100).toFixed(1)}%`,      ""],
    ["F1-Score",           `${(met.f1*100).toFixed(1)}%`,          ""],
    ["Brier Score",        met.brier_score.toFixed(4),             "lower=better"],
    ["Decision Threshold", met.threshold.toFixed(4),               ""],
  ];

  const rows = metricRows.map(([k,v,hint]) =>
    `<div class="mm-metric-row">
       <span class="mm-mr-key">${k}</span>
       <span class="mm-mr-val mm-c-${c}">${v}</span>
       ${hint ? `<span class="mm-mr-hint">${hint}</span>` : ""}
     </div>`
  ).join("");

  let extraBlock = "";
  if (m.architecture && Object.keys(m.architecture).length) {
    const archRows = Object.entries(m.architecture)
      .map(([k,v]) => `<div class="mm-arch-row"><span>${k.replace(/_/g," ")}</span><strong>${v}</strong></div>`)
      .join("");
    extraBlock = `<div class="mm-section-hd" style="margin-top:14px">Architecture</div><div class="mm-arch-grid">${archRows}</div>`;
  } else if (m.feature_count) {
    extraBlock = `<div class="mm-section-hd" style="margin-top:14px">Model Info</div>
      <div class="mm-arch-grid">
        <div class="mm-arch-row"><span>Feature count</span><strong>${m.feature_count}</strong></div>
      </div>`;
  }

  const chartTitle = m.type === "sequence"
    ? "Training Curves (loss + val AUC per epoch)"
    : "Top-10 Feature Importance (Gain)";
  const chartEl = m.type === "sequence"
    ? `<svg id="mm-training-chart" class="mm-svg mm-svg-tall" viewBox="0 0 680 220" preserveAspectRatio="xMidYMid meet"></svg>
       <div class="mm-chart-legend">
         <span class="mm-leg mm-leg-amber-dash">— Train Loss</span>
         <span class="mm-leg mm-leg-teal">— Val Loss</span>
         <span class="mm-leg mm-leg-violet">— Val AUC (right axis)</span>
       </div>`
    : `<svg id="mm-feature-chart" class="mm-svg mm-svg-tall" viewBox="0 0 680 260" preserveAspectRatio="xMidYMid meet"></svg>`;

  return `
    <div class="mm-detail-layout">
      <div class="mm-detail-left">
        <div class="mm-section-hd">${m.name} · Test Set</div>
        <div class="mm-metric-list">${rows}</div>
        ${extraBlock}
      </div>
      <div class="mm-detail-right">
        <div class="mm-section-hd">${chartTitle}</div>
        ${chartEl}
      </div>
    </div>`;
}

// ── Comparison chart (grouped bars) ──────────────────────────────────────
function drawMMComparisonChart() {
  const svg = $("mm-comparison-chart");
  if (!svg || !_metricsData) return;
  const W=760, H=190, padL=36, padR=12, padT=18, padB=46;
  const innerW = W-padL-padR, innerH = H-padT-padB;
  const models = [
    {key:"sepsis_gru", label:"Sepsis GRU", color:MM_COLORS.teal},
    {key:"sepsis_xgb", label:"Sepsis XGB", color:MM_COLORS.amber},
    {key:"resp_gru",   label:"Resp GRU",   color:MM_COLORS.violet},
    {key:"resp_xgb",   label:"Resp XGB",   color:MM_COLORS.blue},
  ];
  const metrics = [{key:"auc",label:"AUC"},{key:"average_precision",label:"AUPRC"},{key:"f1",label:"F1"}];
  const nM = models.length, nK = metrics.length;
  const groupW = innerW / nM;
  const barW   = groupW / (nK + 1.5);
  let out = "";

  // Grid
  for (let i=0; i<=5; i++) {
    const y = padT + innerH * (1 - i/5);
    const v = (i/5).toFixed(1);
    out += `<line x1="${padL}" y1="${y.toFixed(1)}" x2="${W-padR}" y2="${y.toFixed(1)}" stroke="${MM_COLORS.border}" stroke-width="1"/>`;
    out += `<text x="${padL-4}" y="${(y+3).toFixed(1)}" text-anchor="end" font-size="9" fill="${MM_COLORS.text3}">${v}</text>`;
  }

  // Bars
  models.forEach((model, mi) => {
    const met = _metricsData[model.key].metrics;
    metrics.forEach((mk, ki) => {
      const val = met[mk.key];
      const x   = padL + mi*groupW + (ki+0.75)*barW;
      const bh  = val * innerH;
      const y   = padT + innerH - bh;
      const op  = 0.45 + ki * 0.2;
      out += `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${barW.toFixed(1)}" height="${bh.toFixed(1)}" fill="${model.color}" opacity="${op}" rx="2"/>`;
      if (bh > 14) {
        out += `<text x="${(x+barW/2).toFixed(1)}" y="${(y+bh-4).toFixed(1)}" text-anchor="middle" font-size="8" fill="${model.color}" opacity="0.9">${val.toFixed(2)}</text>`;
      } else {
        out += `<text x="${(x+barW/2).toFixed(1)}" y="${(y-3).toFixed(1)}" text-anchor="middle" font-size="8" fill="${model.color}">${val.toFixed(2)}</text>`;
      }
    });
    const lx = padL + mi*groupW + groupW/2;
    out += `<text x="${lx.toFixed(1)}" y="${H-padB+14}" text-anchor="middle" font-size="10" fill="${model.color}" font-weight="600">${model.label}</text>`;
  });

  svg.innerHTML = out;
}

// ── Training curve ────────────────────────────────────────────────────────
function drawMMTrainingCurve(history) {
  const svg = $("mm-training-chart");
  if (!svg || !history?.length) return;
  const W=680, H=220, padL=48, padR=52, padT=14, padB=28;
  const innerW = W-padL-padR, innerH = H-padT-padB;
  const n = history.length;

  const allLoss = history.flatMap(h => [h.train_loss, h.val_loss]);
  const maxL = Math.max(...allLoss), minL = Math.min(...allLoss);
  const rngL = maxL - minL || 0.001;

  const aucs = history.map(h => h.val_auc);
  const maxA = Math.max(...aucs), minA = Math.min(...aucs);
  const rngA = maxA - minA || 0.001;

  const sx  = i => padL + (i / (n-1)) * innerW;
  const syL = v => padT + innerH - ((v-minL)/rngL)*innerH;
  const syA = v => padT + innerH - ((v-minA)/rngA)*innerH;

  let out = "";

  // Grid lines + left axis (loss)
  for (let i=0; i<=4; i++) {
    const y = padT + (i/4)*innerH;
    const v = (maxL - (i/4)*rngL).toFixed(3);
    out += `<line x1="${padL}" y1="${y.toFixed(1)}" x2="${W-padR}" y2="${y.toFixed(1)}" stroke="${MM_COLORS.border}" stroke-width="1"/>`;
    out += `<text x="${padL-4}" y="${(y+3).toFixed(1)}" text-anchor="end" font-size="9" fill="${MM_COLORS.text3}">${v}</text>`;
  }

  // Right axis (val AUC)
  for (let i=0; i<=4; i++) {
    const y = padT + (i/4)*innerH;
    const v = (maxA - (i/4)*rngA).toFixed(3);
    out += `<text x="${W-padR+4}" y="${(y+3).toFixed(1)}" font-size="9" fill="${MM_COLORS.violet}">${v}</text>`;
  }
  out += `<text x="${W-padR+4}" y="${H-padB+22}" font-size="9" fill="${MM_COLORS.violet}">AUC</text>`;

  // Train loss (amber dashed)
  const trainPts = history.map((h,i) => `${i===0?"M":"L"}${sx(i).toFixed(1)},${syL(h.train_loss).toFixed(1)}`).join(" ");
  out += `<path d="${trainPts}" stroke="${MM_COLORS.amber}" stroke-width="1.5" stroke-dasharray="5,3" fill="none"/>`;

  // Val loss (teal solid)
  const valPts = history.map((h,i) => `${i===0?"M":"L"}${sx(i).toFixed(1)},${syL(h.val_loss).toFixed(1)}`).join(" ");
  out += `<path d="${valPts}" stroke="${MM_COLORS.teal}" stroke-width="2" fill="none"/>`;

  // Val AUC (violet solid, right scale)
  const aucPts = history.map((h,i) => `${i===0?"M":"L"}${sx(i).toFixed(1)},${syA(h.val_auc).toFixed(1)}`).join(" ");
  out += `<path d="${aucPts}" stroke="${MM_COLORS.violet}" stroke-width="1.8" fill="none" opacity="0.75"/>`;

  // X-axis epoch labels
  const step = Math.max(1, Math.floor(n/7));
  for (let i=0; i<n; i+=step) {
    out += `<text x="${sx(i).toFixed(1)}" y="${H-padB+14}" text-anchor="middle" font-size="9" fill="${MM_COLORS.text3}">${history[i].epoch}</text>`;
  }
  out += `<text x="${(padL+innerW/2).toFixed(1)}" y="${H-2}" text-anchor="middle" font-size="9" fill="${MM_COLORS.text3}">Epoch</text>`;
  out += `<text x="${padL-36}" y="${(padT+innerH/2).toFixed(1)}" text-anchor="middle" font-size="9" fill="${MM_COLORS.text3}" transform="rotate(-90,${padL-36},${(padT+innerH/2).toFixed(1)})">Loss</text>`;

  svg.innerHTML = out;
}

// ── Feature importance chart ──────────────────────────────────────────────
function drawMMFeatureChart(features) {
  const svg = $("mm-feature-chart");
  if (!svg || !features?.length) return;
  const W=680, H=260, padL=170, padR=58, padT=8, padB=8;
  const innerW = W-padL-padR, innerH = H-padT-padB;
  const n = features.length;
  const rowH  = Math.floor(innerH / n);
  const barH  = Math.max(8, rowH - 5);
  let out = "";

  features.forEach((f, i) => {
    const y   = padT + i*rowH + (rowH-barH)/2;
    const bw  = f.rel * innerW;
    const op  = 0.4 + f.rel * 0.6;
    const lbl = f.feature.replace(/__\w+$/, "").replace(/_/g," ").substring(0,24);
    const gainStr = f.gain >= 1000 ? `${(f.gain/1000).toFixed(1)}k` : f.gain.toFixed(0);
    const rank = i+1;

    out += `<text x="${padL-8}" y="${(y+barH/2+3.5).toFixed(1)}" text-anchor="end" font-size="9.5" fill="${MM_COLORS.text2}">${lbl}</text>`;
    out += `<text x="${padL-8}" y="${(y+barH/2+13).toFixed(1)}" text-anchor="end" font-size="8" fill="${MM_COLORS.text3}" opacity="0.6">#${rank}</text>`;
    out += `<rect x="${padL}" y="${y.toFixed(1)}" width="${innerW}" height="${barH}" fill="${MM_COLORS.border}" rx="3"/>`;
    out += `<rect x="${padL}" y="${y.toFixed(1)}" width="${bw.toFixed(1)}" height="${barH}" fill="${MM_COLORS.teal}" opacity="${op}" rx="3"/>`;
    out += `<text x="${(padL+bw+5).toFixed(1)}" y="${(y+barH/2+3.5).toFixed(1)}" font-size="9" fill="${MM_COLORS.text2}">${gainStr}</text>`;
  });

  svg.innerHTML = out;
}

// ── Boot ──────────────────────────────────────────────────────────────────
async function boot() {
  startClock();
  loadAlertHistory();

  const IDS = loadBoardIds();

  // Populate state instantly, render empty cards
  for (const id of IDS) state.patients.set(id, {id, window:null, result:null, loadedAt:null, loading:false});
  renderGrid();

  // Health check (background)
  refreshHealth();

  // Load first patient fully (window + evaluate) — shows results fast
  await loadPatient(IDS[0], true);

  // Load remaining windows in parallel (fast — file reads only)
  await Promise.all(IDS.slice(1).map(id => loadPatientData(id)));

  // Evaluate remaining patients in parallel
  await Promise.all(IDS.slice(1).map(id => evaluatePatient(id)));

  // Start live monitoring for all patients (vital tick + frame advance)
  startLiveMonitoring();
}

// ── Event bindings ────────────────────────────────────────────────────────
$("load-demo")?.addEventListener("click", () => {
  const id = patientInput?.value?.trim();
  if (!id) {
    if (patientInput) { patientInput.style.outline = "1px solid var(--red)"; setTimeout(() => { patientInput.style.outline = ""; }, 1200); }
    return;
  }
  if (!state.patients.has(id)) loadPatient(id, true);
  else selectPatient(id);
});
patientInput?.addEventListener("keydown", e => { if (e.key==="Enter") $("load-demo")?.click(); });
$("model-metrics-btn")?.addEventListener("click", openMetricsModal);
$("mm-close")?.addEventListener("click", closeMetricsModal);
$("metrics-modal-backdrop")?.addEventListener("click", e => { if (e.target === $("metrics-modal-backdrop")) closeMetricsModal(); });
document.querySelectorAll(".mm-tab").forEach(btn => btn.addEventListener("click", () => renderMetricsTab(btn.dataset.tab)));
$("browse-patients-btn")?.addEventListener("click", openPatientModal);
$("reset-board-btn")?.addEventListener("click", () => {
  try { localStorage.setItem(BOARD_KEY, JSON.stringify(DEFAULT_IDS)); } catch(_) {}
  location.reload();
});
$("pm-close")?.addEventListener("click", closePatientModal);
$("patient-modal-backdrop")?.addEventListener("click", e => {
  if (e.target === $("patient-modal-backdrop")) closePatientModal();
});
$("pm-search")?.addEventListener("input", e => pmDebounceSearch(e.target.value));
document.querySelectorAll(".pm-filter").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".pm-filter").forEach(b => b.classList.toggle("active", b === btn));
    // "Not on board" is a local filter — re-render without API call
    if (btn.dataset.filter === "new") {
      renderPatientRows(pmResults.filter(p => !state.patients.has(p.id)));
    } else {
      fetchPatients();
    }
  });
});
document.addEventListener("keydown", e => { if (e.key === "Escape") { closePatientModal(); closeMetricsModal(); } });
$("toggle-playback")?.addEventListener("click", reEvaluate);
$("refresh-health")?.addEventListener("click", refreshHealth);
$("close-sidebar")?.addEventListener("click", () => {
  state.selectedId = null;
  if (detailSidebar) detailSidebar.classList.add("hidden");
  renderGrid();
});
$("run-timeline")?.addEventListener("click", () => { if (state.selectedId) runTimelineAnalysis(state.selectedId); });
$("clear-alert-history")?.addEventListener("click", () => {
  state.alertHistory = [];
  try { sessionStorage.removeItem("icu_alerts"); } catch(_) {}
  renderAlertHistory(); updateAlertCounter();
});
$("grid-cols")?.addEventListener("change", e => applyGridCols(e.target.value));
let _stepReqId = 0;
function _doStep(delta) {
  const pid = state.selectedId;
  const e = pid ? state.patients.get(pid) : null;
  if (!e?.window) return;
  const n = Math.min(e.window.length, Math.max(1, (state.monitorIndex || e.window.length) + delta));
  state.monitorIndex = n;
  const reqId = ++_stepReqId;
  apiEvaluate(pid, e.window.slice(0, n)).then(r => {
    // Discard result if a newer step fired or user switched patients
    if (reqId !== _stepReqId || state.selectedId !== pid) return;
    state.patients.set(pid, { ...state.patients.get(pid), result: r, loadedAt: new Date() });
    renderGrid(); renderSidebar(pid);
  }).catch(err => {
    if (err.name !== "AbortError") console.error("[step]:", err);
  });
}
$("step-back")?.addEventListener("click", () => _doStep(-1));
$("step-forward")?.addEventListener("click", () => _doStep(+1));
$("run-eval")?.addEventListener("click", reEvaluate);

// Pause live evaluation (not vital tick) when tab is hidden — resumes on visibility
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) {
    // Re-evaluate all patients that haven't had a fresh result in the last 30s
    const staleMs = 30_000;
    const now = Date.now();
    for (const [id, entry] of state.patients) {
      if (entry.window?.length && (!entry.loadedAt || (now - entry.loadedAt.getTime()) > staleMs)) {
        evaluatePatientSilent(id);
      }
    }
  }
});

boot();

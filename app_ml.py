from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from risk_engine_ml import TriageML, evaluate_patient, MODEL_VERSION

APP_VERSION = "v0.5"
ROOT = Path(__file__).resolve().parent
DEMO_CSV = ROOT / "data" / "demo_patients.csv"
EVENTS_CSV = ROOT / "data" / "demo_events.csv"
SURGE_CSV = ROOT / "data" / "surge_patients.csv"
MANUAL_CSV = ROOT / "data" / "manual_patients.csv"
METRICS_JSON = ROOT / "model" / "metrics.json"
IMPORTANCE_CSV = ROOT / "model" / "feature_importance.csv"

st.set_page_config(
    page_title=f"PatientTriage.ai {APP_VERSION}",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root{
  --pt:#8b00d6;--pt2:#5d00a8;--navy:#111827;--muted:#667085;--line:#e6e8ee;
  --lav:#f7f1fb;--red:#b42318;--redbg:#fff1f0;--amber:#b54708;--amberbg:#fff7e8;
  --green:#067647;--greenbg:#ecfdf3;--blue:#175cd3;--bluebg:#eff8ff;
  --card-bg:#0f1117;--card-line:#2d3139;--card-text:#f3f4f6;--card-muted:#9ca3af;
}
.block-container{padding-top:1.05rem;padding-bottom:2rem;max-width:1500px}
[data-testid="stSidebar"]{background:#0f1117;border-right:1px solid var(--card-line)}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4{color:#f1f5f9}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] [data-testid="stExpander"] summary,
[data-testid="stSidebar"] [data-testid="stExpander"] summary p,
[data-testid="stSidebar"] [data-testid="stExpander"] li{color:#cbd5e1}
[data-testid="stSidebar"] [data-testid="stAlert"] [data-testid="stMarkdownContainer"] p{color:inherit}
h1,h2,h3{color:var(--navy)}
.pt-kicker{font-size:.72rem;letter-spacing:.08em;font-weight:800;color:var(--pt);text-transform:uppercase}
.pt-title{font-size:1.85rem;line-height:1.12;font-weight:850;color:#f1f5f9;margin:.16rem 0 .18rem}
.pt-sub{font-size:.93rem;color:#94a3b8;margin-bottom:.65rem}
.pt-note{font-size:.78rem;color:var(--muted)}
.pt-card{background:var(--card-bg);border:1px solid var(--card-line);border-radius:14px;padding:13px 15px;height:100%}
.pt-card h4{font-size:.78rem;color:var(--card-muted);text-transform:uppercase;letter-spacing:.04em;margin:0 0 6px}
.pt-card .big{font-size:1.35rem;font-weight:850;color:var(--card-text);line-height:1.1}
.pt-card .small{font-size:.80rem;color:var(--card-muted);margin-top:4px}
.pt-alert-red{border-left:5px solid var(--red);background:var(--redbg);padding:10px 12px;border-radius:8px;color:#7a271a}
.pt-alert-amber{border-left:5px solid #f79009;background:var(--amberbg);padding:10px 12px;border-radius:8px;color:#7a2e0e}
.pt-alert-green{border-left:5px solid #12b76a;background:var(--greenbg);padding:10px 12px;border-radius:8px;color:#05603a}
.pt-section{font-size:.78rem;font-weight:800;color:var(--pt);letter-spacing:.05em;text-transform:uppercase;margin:.25rem 0 .55rem}
.pt-priority{display:inline-block;border-radius:999px;padding:4px 9px;font-size:.76rem;font-weight:800;color:white;background:#6941c6}
.pt-footer{margin-top:.75rem;padding:.58rem .85rem;background:var(--lav);border-radius:10px;color:#5d00a8;font-size:.78rem;font-weight:700}
div[data-testid="stMetric"]{background:var(--card-bg);border:1px solid var(--card-line);padding:10px 12px;border-radius:12px}
div[data-testid="stMetric"] [data-testid="stMetricLabel"],
div[data-testid="stMetric"] [data-testid="stMetricLabel"] p{color:var(--card-muted)!important}
div[data-testid="stMetric"] [data-testid="stMetricValue"],
div[data-testid="stMetric"] [data-testid="stMetricValue"] div{color:var(--card-text)!important}
div[data-testid="stMetric"] [data-testid="stMetricDelta"]{color:var(--card-muted)!important}
.stTabs [data-baseweb="tab-list"]{gap:4px}
.stTabs [data-baseweb="tab"]{height:42px;padding:0 12px}
</style>
""",
    unsafe_allow_html=True,
)

ENGINE = TriageML()
HX_COLS = [c for c in ENGINE.features if c.startswith("hx_")]
CORE_EVENT_COLS = [
    "heart_rate", "respiratory_rate", "spo2", "systolic_bp", "diastolic_bp",
    "temperature_c", "spo2_signal_quality", "history_available",
]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def load_demo() -> pd.DataFrame:
    df = _read_csv(DEMO_CSV)
    if not df.empty:
        df["patient_source"] = "demo"
    return df


def load_surge() -> pd.DataFrame:
    df = _read_csv(SURGE_CSV)
    if not df.empty:
        df["patient_source"] = "surge"
    return df


def load_events() -> pd.DataFrame:
    return _read_csv(EVENTS_CSV)


def load_manual() -> pd.DataFrame:
    df = _read_csv(MANUAL_CSV)
    if not df.empty:
        df["patient_source"] = "manual"
    return df


def persist_manual_rows(df: pd.DataFrame) -> None:
    manual = df[df.get("patient_source", pd.Series(index=df.index, dtype=str)).astype(str) == "manual"].copy()
    manual.to_csv(MANUAL_CSV, index=False)


def source_population() -> pd.DataFrame:
    parts = [x for x in [load_demo(), load_manual()] if not x.empty]
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True, sort=False)


def reset_runtime(clear_manual: bool = False) -> None:
    if clear_manual and MANUAL_CSV.exists():
        MANUAL_CSV.unlink()
    st.session_state.current_df = source_population()
    st.session_state.previous_snapshots = {}
    st.session_state.elapsed_since_prev = {}
    st.session_state.sim_minute = 0
    st.session_state.surge_mode = False
    st.session_state.audit_log = []
    st.session_state.applied_events = set()
    st.session_state.removed_live_ids = set()


def sync_demo_source() -> None:
    """Keep CSV-backed demo membership synchronized without overwriting live updated vitals.

    If a row is deleted from demo_patients.csv it disappears from the dashboard on the next rerun.
    New demo IDs are added. Existing demo rows retain any repeat-vital changes already applied in the session.
    Manual patients are stored separately and are not affected by demo CSV edits.
    """
    current = st.session_state.current_df.copy()
    demo = load_demo()
    source_ids = set(demo.get("demo_id", pd.Series(dtype=str)).astype(str))
    removed_live = set(st.session_state.removed_live_ids)

    if not current.empty and "patient_source" in current.columns:
        cur_demo = current["patient_source"].astype(str).eq("demo")
        drop_mask = cur_demo & ~current["demo_id"].astype(str).isin(source_ids)
        current = current.loc[~drop_mask].copy()

    current_ids = set(current.get("demo_id", pd.Series(dtype=str)).astype(str))
    to_add = demo[~demo["demo_id"].astype(str).isin(current_ids | removed_live)] if not demo.empty else demo
    if not to_add.empty:
        current = pd.concat([current, to_add], ignore_index=True, sort=False)

    st.session_state.current_df = current.reset_index(drop=True)


def age_group_for(age: int) -> str:
    if age <= 16:
        return "pediatric"
    if age <= 39:
        return "young_adult"
    if age <= 64:
        return "middle_aged"
    return "elderly"


def bool_or_false(v) -> bool:
    if pd.isna(v):
        return False
    return bool(v)


def eval_population(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        pid = str(row["demo_id"])
        prev_dict = st.session_state.previous_snapshots.get(pid)
        prev = pd.Series(prev_dict) if prev_dict else None
        elapsed = st.session_state.elapsed_since_prev.get(pid)
        ev = evaluate_patient(ENGINE, row, prev, elapsed)
        rec = ev.as_dict()
        rec.update({
            "demo_id": pid,
            "patient_source": row.get("patient_source", "demo"),
            "source_patient_id": row.get("source_patient_id", row.get("patient_id")),
            "age": row.get("age"),
            "age_group": row.get("age_group"),
            "sex": row.get("sex"),
            "chief_complaint_raw": row.get("chief_complaint_raw", ""),
            "scenario": row.get("scenario", ""),
            "wait_minutes": row.get("wait_minutes", 0),
            "history_available": row.get("history_available", True),
        })
        records.append(rec)
    out = pd.DataFrame(records)
    if out.empty:
        return out
    out = out.sort_values(["attention_score", "recommended_priority"], ascending=[False, True]).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def advance_clock(minutes: int = 10, apply_scripted_events: bool = True) -> list[str]:
    st.session_state.sim_minute += int(minutes)
    df = st.session_state.current_df.copy()
    if not df.empty:
        df["wait_minutes"] = pd.to_numeric(df.get("wait_minutes", 0), errors="coerce").fillna(0) + minutes
        df["last_assessed_minutes"] = pd.to_numeric(df.get("last_assessed_minutes", 0), errors="coerce").fillna(0) + minutes

    applied_notes: list[str] = []
    if apply_scripted_events:
        events = load_events()
        if not events.empty:
            due = events[events["event_minute"] == st.session_state.sim_minute]
            for _, event in due.iterrows():
                pid = str(event["demo_id"])
                key = (pid, int(event["event_minute"]))
                if key in st.session_state.applied_events:
                    continue
                mask = df.get("demo_id", pd.Series(dtype=str)).astype(str).eq(pid)
                if not mask.any():
                    continue
                idx = df.index[mask][0]
                old = df.loc[idx].copy()
                elapsed = float(pd.to_numeric(pd.Series([old.get("last_assessed_minutes", minutes)]), errors="coerce").fillna(minutes).iloc[0])
                elapsed = max(1.0, elapsed)
                st.session_state.previous_snapshots[pid] = old.to_dict()
                st.session_state.elapsed_since_prev[pid] = elapsed
                for c in CORE_EVENT_COLS:
                    if c in event.index and pd.notna(event[c]):
                        df.at[idx, c] = event[c]
                df.at[idx, "last_assessed_minutes"] = 0
                st.session_state.applied_events.add(key)
                applied_notes.append(f"{pid}: {event.get('note', 'repeat observation applied')}")

    st.session_state.current_df = df
    # Keep manual-patient latest state durable across restarts.
    persist_manual_rows(df)
    return applied_notes


def remove_from_live_list(pid: str, reason: str) -> None:
    df = st.session_state.current_df.copy()
    row = df[df["demo_id"].astype(str) == pid]
    if row.empty:
        return
    source = str(row.iloc[0].get("patient_source", "demo"))
    st.session_state.current_df = df[df["demo_id"].astype(str) != pid].reset_index(drop=True)
    if source == "demo":
        st.session_state.removed_live_ids.add(pid)
    persist_manual_rows(st.session_state.current_df)
    st.session_state.audit_log.append({
        "sim_minute": st.session_state.sim_minute,
        "patient": pid,
        "action": "Removed from live list",
        "reason": reason,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    })


def patient_row(pid: str, active: pd.DataFrame) -> pd.Series:
    return active[active["demo_id"].astype(str) == str(pid)].iloc[0]


def render_priority_card(rec: pd.Series) -> None:
    level = int(rec["recommended_priority"])
    color = {1: "#b42318", 2: "#d92d20", 3: "#f79009", 4: "#175cd3", 5: "#067647"}[level]
    alert = str(rec["alert"])
    st.markdown(
        f"""
<div class="pt-card" style="border-top:4px solid {color}">
  <h4>#{int(rec['rank'])} · {rec['demo_id']} · Level {level}</h4>
  <div class="big">{str(rec['chief_complaint_raw'])[:74]}</div>
  <div class="small">Age {int(rec['age'])} · {rec['age_group']} · Wait {int(float(rec['wait_minutes']))} min</div>
  <div class="small"><b>{rec['trajectory']}</b> · {rec['uncertainty']} uncertainty · {rec['freshness']}</div>
  <div class="small" style="color:{color};font-weight:700;margin-top:7px">{alert}</div>
</div>
""",
        unsafe_allow_html=True,
    )


if "current_df" not in st.session_state:
    reset_runtime(clear_manual=False)
else:
    sync_demo_source()

# ---------------- Sidebar: demo and operational controls ----------------
st.sidebar.markdown("### ED controls")
st.sidebar.caption(f"Prototype {APP_VERSION} · model {MODEL_VERSION}")
st.sidebar.info("Research/demo software only. Synthetic Triagegeist benchmark; not for clinical care.")

c1, c2 = st.sidebar.columns(2)
if c1.button("+10 min", use_container_width=True, type="primary", help="Advances the simulated ED clock and applies any scripted repeat observations due at that time."):
    notes = advance_clock(10, apply_scripted_events=True)
    st.session_state.last_event_notes = notes
    st.rerun()
if c2.button("Reload", use_container_width=True, help="Reloads demo/manual sources and clears temporal state."):
    reset_runtime(clear_manual=False)
    st.rerun()

st.session_state.surge_mode = st.sidebar.toggle("3× surge simulation", value=bool(st.session_state.surge_mode))

with st.sidebar.expander("Demo scenario guide", expanded=True):
    st.markdown("""
**+10 min**  
• **P014** respiratory deterioration begins  
• **P016** geriatric respiratory case becomes critical  
• **P006** missing BP / poor signal are resolved  
• **P009** suspect low SpO₂ is verified as measurement artifact

**+20 min**  
• **P014** deteriorates further → strong ML Risk Delta  
• **P003** pediatric case worsens  
• **P017** subtle geriatric case deteriorates  
• **P011** high-risk chest-pain repeat remains stable

**+30 min**  
• **P020** immunosuppressed infection worsens  
• **P004** low-risk pediatric repeat remains stable
""")

if st.sidebar.button("Clear manually added patients", use_container_width=True):
    if MANUAL_CSV.exists():
        MANUAL_CSV.unlink()
    cur = st.session_state.current_df
    if not cur.empty:
        st.session_state.current_df = cur[cur.get("patient_source", "demo").astype(str) != "manual"].reset_index(drop=True)
    st.rerun()

last_notes = st.session_state.pop("last_event_notes", []) if "last_event_notes" in st.session_state else []
if last_notes:
    with st.sidebar.expander("Repeat observations just applied", expanded=True):
        for n in last_notes:
            st.caption(n)

# ---------------- Live population ----------------
active = st.session_state.current_df.copy()
if st.session_state.surge_mode:
    surge = load_surge()
    if not surge.empty:
        active = pd.concat([active, surge], ignore_index=True, sort=False)
queue = eval_population(active)

# ---------------- Header ----------------
st.markdown('<div class="pt-kicker">Emergency Department · Live Decision Support</div>', unsafe_allow_html=True)
st.markdown('<div class="pt-title">PatientTriage.ai — Continuous Clinical Attention Board</div>', unsafe_allow_html=True)
st.markdown('<div class="pt-sub">Acuity prediction, uncertainty, repeat-vital risk delta and reassessment freshness — with final decisions retained by clinical staff.</div>', unsafe_allow_html=True)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Active", len(active), "3× surge" if st.session_state.surge_mode else "Normal load")
m2.metric("Level 1–2", int((queue["recommended_priority"] <= 2).sum()) if not queue.empty else 0)
m3.metric("Deteriorating", int(queue["trajectory"].astype(str).isin(["Worsening", "Rapidly worsening"]).sum()) if not queue.empty else 0)
m4.metric("Manual review", int(queue["manual_review"].sum()) if not queue.empty else 0)
m5.metric("Overdue", int(queue["freshness"].astype(str).str.contains("overdue", case=False).sum()) if not queue.empty else 0)

st.markdown('<div class="pt-footer">Queue order is an attention-ranking aid, not an autonomous treatment decision. Uncertainty can trigger review but never downgrades a patient.</div>', unsafe_allow_html=True)

# ---------------- Tabs ----------------
board_tab, review_tab, add_tab, repeat_tab, action_tab, validation_tab = st.tabs([
    "ED Live Board", "Patient Review", "Add Patient", "Record Repeat Vitals", "Clinician Actions", "Technical Validation"
])

# ====================== BOARD ======================
with board_tab:
    st.markdown("### Live Safety Queue")
    if queue.empty:
        st.warning("No active patients. Add a patient or restore demo_patients.csv.")
    else:
        # Most important information first.
        st.markdown('<div class="pt-section">Immediate attention</div>', unsafe_allow_html=True)
        top = queue.head(min(4, len(queue)))
        cols = st.columns(len(top))
        for col, (_, rec) in zip(cols, top.iterrows()):
            with col:
                render_priority_card(rec)

        st.markdown('<div class="pt-section" style="margin-top:1rem">All active patients</div>', unsafe_allow_html=True)
        f1, f2, f3 = st.columns([1.2, 1, 1])
        search = f1.text_input("Search", placeholder="Patient ID or complaint", label_visibility="collapsed")
        priority_filter = f2.multiselect("Priority", [1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5], label_visibility="collapsed")
        alerts_only = f3.toggle("Alerts / review only", value=False)

        showq = queue[queue["recommended_priority"].isin(priority_filter)].copy()
        if search:
            s = search.lower()
            showq = showq[
                showq["demo_id"].astype(str).str.lower().str.contains(s, regex=False)
                | showq["chief_complaint_raw"].astype(str).str.lower().str.contains(s, regex=False)
            ]
        if alerts_only:
            showq = showq[(showq["alert"] != "No active alert") | showq["manual_review"]]

        board = showq[[
            "rank", "demo_id", "age", "chief_complaint_raw", "recommended_priority",
            "trajectory", "uncertainty", "freshness", "wait_minutes", "urgent_probability", "alert"
        ]].copy()
        board["urgent_probability"] = (100 * board["urgent_probability"]).round(1)
        board["wait_minutes"] = pd.to_numeric(board["wait_minutes"], errors="coerce").fillna(0).astype(int)
        board.columns = ["Rank", "Patient", "Age", "Chief complaint", "Level", "Trend", "Uncertainty", "Freshness", "Wait min", "Urgent risk %", "Alert"]
        st.dataframe(
            board,
            use_container_width=True,
            hide_index=True,
            height=520,
            column_config={
                "Rank": st.column_config.NumberColumn(width="small"),
                "Patient": st.column_config.TextColumn(width="small"),
                "Age": st.column_config.NumberColumn(width="small"),
                "Level": st.column_config.NumberColumn(width="small"),
                "Wait min": st.column_config.NumberColumn(width="small"),
                "Urgent risk %": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%", width="medium"),
                "Chief complaint": st.column_config.TextColumn(width="large"),
                "Alert": st.column_config.TextColumn(width="large"),
            },
        )
        st.caption("Demo CSV membership is synchronized on every rerun: delete a patient row from data/demo_patients.csv and that patient disappears from the dashboard. Manually added patients are stored separately.")

# ====================== REVIEW ======================
with review_tab:
    st.markdown("### Patient Review")
    if queue.empty:
        st.info("No active patients.")
    else:
        ids = queue["demo_id"].astype(str).tolist()
        default_pid = "P014" if "P014" in ids else ids[0]
        pid = st.selectbox("Patient", ids, index=ids.index(default_pid), key="review_pid")
        row = patient_row(pid, active)
        prev_dict = st.session_state.previous_snapshots.get(pid)
        prev = pd.Series(prev_dict) if prev_dict else None
        elapsed = st.session_state.elapsed_since_prev.get(pid)
        ev = evaluate_patient(ENGINE, row, prev, elapsed)

        left_title, right_title = st.columns([4, 1])
        with left_title:
            st.markdown(f"#### {pid} · {row.get('chief_complaint_raw', '')}")
            st.caption(f"Age {int(row['age'])} · {row.get('age_group')} · {row.get('sex')} · {row.get('arrival_mode')} · {row.get('scenario', 'Live patient')}")
        with right_title:
            st.markdown(f"<div style='text-align:right'><span class='pt-priority'>Recommended Level {ev.recommended_priority}</span></div>", unsafe_allow_html=True)

        if "CRITICAL" in ev.alert or "RAPID" in ev.alert:
            st.markdown(f'<div class="pt-alert-red"><b>{ev.alert}</b></div>', unsafe_allow_html=True)
        elif ev.alert != "No active alert":
            st.markdown(f'<div class="pt-alert-amber"><b>{ev.alert}</b></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="pt-alert-green"><b>No active alert.</b> Continue on the reassessment clock.</div>', unsafe_allow_html=True)

        st.markdown('<div class="pt-section">Risk envelope</div>', unsafe_allow_html=True)
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Current risk", ev.current_risk, f"Level {ev.recommended_priority}")
        e2.metric("Trajectory", ev.trajectory, f"Δ urgent {ev.ml_risk_delta*100:+.1f} pp" if elapsed else "Initial assessment")
        e3.metric("Uncertainty", ev.uncertainty, f"Model confidence {ev.confidence:.1f}%")
        e4.metric("Freshness", ev.freshness, f"Wait {int(float(row.get('wait_minutes', 0) or 0))} min")

        st.markdown('<div class="pt-section">Current observations</div>', unsafe_allow_html=True)
        v1, v2, v3, v4, v5, v6 = st.columns(6)
        def metric_value(v, suffix=""):
            return "—" if pd.isna(v) else f"{float(v):.1f}{suffix}"
        v1.metric("HR", metric_value(row.get("heart_rate")))
        v2.metric("BP", "—" if pd.isna(row.get("systolic_bp")) else f"{float(row['systolic_bp']):.0f}/{float(row['diastolic_bp']):.0f}")
        v3.metric("RR", metric_value(row.get("respiratory_rate")))
        v4.metric("SpO₂", metric_value(row.get("spo2"), "%"))
        v5.metric("Temp", metric_value(row.get("temperature_c"), "°C"))
        v6.metric("GCS", metric_value(row.get("gcs_total")))

        lcol, rcol = st.columns([1.05, 0.95])
        with lcol:
            st.markdown("#### Acuity probability distribution")
            pvals = [ev.probability_esi1, ev.probability_esi2, ev.probability_esi3, ev.probability_esi4, ev.probability_esi5]
            fig = go.Figure(go.Bar(
                x=["Level 1", "Level 2", "Level 3", "Level 4", "Level 5"],
                y=[v * 100 for v in pvals],
                text=[f"{v*100:.1f}%" for v in pvals],
                textposition="outside",
                marker_color=["#b42318", "#d92d20", "#f79009", "#175cd3", "#067647"],
            ))
            fig.update_layout(height=310, margin=dict(l=10, r=10, t=15, b=15), yaxis_title="Probability %", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            if ev.physiological_delta:
                delta_text = " · ".join([f"{k}: {v:+.1f}/10 min" for k, v in ev.physiological_delta.items()])
                st.caption(f"Repeat-vital change: {delta_text}")
        with rcol:
            st.markdown("#### What needs attention")
            st.info(ev.next_best_information)
            if ev.discordance != "None":
                st.warning(ev.discordance)
            st.markdown("**Model associations**")
            for drv in ev.top_model_drivers[:5]:
                st.write(f"• {drv}")
            hist = bool_or_false(row.get("history_available", True))
            if not hist:
                st.warning("Prior health record unavailable. History features are treated as unknown, not as absence of disease.")
            else:
                active_hx = [c.replace("hx_", "").replace("_", " ") for c in HX_COLS if bool_or_false(row.get(c))]
                if active_hx:
                    st.caption("Recorded history: " + ", ".join(active_hx[:8]) + ("…" if len(active_hx) > 8 else ""))
                else:
                    st.caption("Prior record is available; no listed history flag is active in the demo record.")

# ====================== ADD PATIENT ======================
with add_tab:
    st.markdown("### Add Patient")
    st.caption("Enter triage-time information only. Unknown information can be left unavailable; the model handles missing values and the uncertainty layer will surface them.")

    with st.form("add_patient_form", clear_on_submit=False):
        st.markdown('<div class="pt-section">1 · Patient & arrival</div>', unsafe_allow_html=True)
        a1, a2, a3, a4 = st.columns(4)
        suggested_id = f"N{int(datetime.now(timezone.utc).timestamp()) % 100000:05d}"
        new_id = a1.text_input("Patient ID", value=suggested_id)
        age = int(a2.number_input("Age", min_value=0, max_value=110, value=45, step=1))
        sex = a3.selectbox("Sex", ENGINE.category_levels["sex"], index=0)
        arrival_mode = a4.selectbox("Arrival mode", [x for x in ENGINE.category_levels["arrival_mode"] if x != "UNKNOWN"])
        b1, b2 = st.columns(2)
        transport_origin = b1.selectbox("Origin", [x for x in ENGINE.category_levels["transport_origin"] if x != "UNKNOWN"])
        wait_minutes = int(b2.number_input("Already waiting (min)", min_value=0, max_value=240, value=0, step=1))

        st.markdown('<div class="pt-section">2 · Presentation</div>', unsafe_allow_html=True)
        chief_raw = st.text_input("Chief complaint / nurse-facing description", placeholder="e.g., chest tightness with shortness of breath")
        p1, p2, p3 = st.columns(3)
        chief_system = p1.selectbox("Complaint system", [x for x in ENGINE.category_levels["chief_complaint_system"] if x != "UNKNOWN"])
        pain_location = p2.selectbox("Pain location", [x for x in ENGINE.category_levels["pain_location"] if x != "UNKNOWN"])
        mental = p3.selectbox("Mental status", [x for x in ENGINE.category_levels["mental_status_triage"] if x != "UNKNOWN"])

        st.markdown('<div class="pt-section">3 · Current observations</div>', unsafe_allow_html=True)
        bp_available = st.checkbox("Blood pressure available", value=True)
        c1, c2, c3, c4 = st.columns(4)
        hr = float(c1.number_input("Heart rate", min_value=20.0, max_value=240.0, value=85.0, step=1.0))
        rr = float(c2.number_input("Respiratory rate", min_value=4.0, max_value=60.0, value=18.0, step=1.0))
        spo2 = float(c3.number_input("SpO₂ %", min_value=50.0, max_value=100.0, value=98.0, step=0.1))
        temp = float(c4.number_input("Temperature °C", min_value=30.0, max_value=43.0, value=37.0, step=0.1))
        d1, d2, d3, d4 = st.columns(4)
        sbp = float(d1.number_input("Systolic BP", min_value=40.0, max_value=260.0, value=125.0, step=1.0, disabled=not bp_available)) if bp_available else np.nan
        dbp = float(d2.number_input("Diastolic BP", min_value=20.0, max_value=180.0, value=75.0, step=1.0, disabled=not bp_available)) if bp_available else np.nan
        gcs = float(d3.number_input("GCS", min_value=3.0, max_value=15.0, value=15.0, step=1.0))
        pain = float(d4.number_input("Pain score", min_value=0.0, max_value=10.0, value=3.0, step=1.0))
        q1, q2, q3 = st.columns(3)
        signal_quality = q1.selectbox("SpO₂ signal quality", ["good", "noisy", "poor", "unreliable"])
        body_available = q2.checkbox("Height / weight available", value=False)
        news_available = q3.checkbox("Protocol score available", value=False, help="Optional empirical comparator; not used as an ML predictor.")
        body1, body2, body3 = st.columns(3)
        weight = float(body1.number_input("Weight kg", min_value=1.0, max_value=300.0, value=70.0, disabled=not body_available)) if body_available else np.nan
        height = float(body2.number_input("Height cm", min_value=40.0, max_value=230.0, value=170.0, disabled=not body_available)) if body_available else np.nan
        news2 = float(body3.number_input("NEWS2 / local score", min_value=0.0, max_value=20.0, value=0.0, disabled=not news_available)) if news_available else np.nan

        st.markdown('<div class="pt-section">4 · Prior health record</div>', unsafe_allow_html=True)
        history_available = st.toggle("Prior health record available", value=True)
        if history_available:
            h1, h2, h3 = st.columns(3)
            prior_ed = int(h1.number_input("ED visits in past 12m", min_value=0, max_value=50, value=0, step=1))
            prior_adm = int(h2.number_input("Admissions in past 12m", min_value=0, max_value=30, value=0, step=1))
            active_meds = int(h3.number_input("Active medications", min_value=0, max_value=50, value=0, step=1))
            with st.expander("Detailed history used by the ML model", expanded=False):
                hx_values = {}
                hx_labels = [c.replace("hx_", "").replace("_", " ").title() for c in HX_COLS]
                hx_grid = st.columns(3)
                for i, (colname, label) in enumerate(zip(HX_COLS, hx_labels)):
                    hx_values[colname] = 1.0 if hx_grid[i % 3].checkbox(label, value=False, key=f"add_{colname}") else 0.0
        else:
            prior_ed = prior_adm = active_meds = np.nan
            hx_values = {c: np.nan for c in HX_COLS}
            st.info("No prior record: history variables will be stored as unknown (NaN), not as 'no disease'.")

        submitted = st.form_submit_button("Triage & add to live board", type="primary", use_container_width=True)
        if submitted:
            existing = set(st.session_state.current_df.get("demo_id", pd.Series(dtype=str)).astype(str))
            if not new_id.strip():
                st.error("Patient ID is required.")
            elif new_id.strip() in existing:
                st.error("Patient ID already exists on the live board.")
            elif not chief_raw.strip():
                st.error("Chief complaint is required.")
            else:
                record = {
                    "demo_id": new_id.strip(), "patient_source": "manual", "source_patient_id": new_id.strip(),
                    "scenario": "Manually entered live patient", "history_available": history_available,
                    "spo2_signal_quality": signal_quality, "last_assessed_minutes": 0, "wait_minutes": wait_minutes,
                    "chief_complaint_raw": chief_raw.strip(), "arrival_mode": arrival_mode, "age": age,
                    "age_group": age_group_for(age), "sex": sex, "transport_origin": transport_origin,
                    "pain_location": pain_location, "mental_status_triage": mental, "chief_complaint_system": chief_system,
                    "num_prior_ed_visits_12m": prior_ed, "num_prior_admissions_12m": prior_adm,
                    "num_active_medications": active_meds,
                    "num_comorbidities": float(sum(v == 1.0 for v in hx_values.values())) if history_available else np.nan,
                    "systolic_bp": sbp, "diastolic_bp": dbp, "heart_rate": hr, "respiratory_rate": rr,
                    "temperature_c": temp, "spo2": spo2, "gcs_total": gcs, "pain_score": pain,
                    "weight_kg": weight, "height_cm": height, "news2_score": news2,
                }
                record.update(hx_values)
                # Derived fields are recomputed by the engine; keep placeholders in the stored row.
                for c in ["mean_arterial_pressure", "pulse_pressure", "bmi", "shock_index"]:
                    record[c] = np.nan
                new_row = pd.DataFrame([record])
                st.session_state.current_df = pd.concat([st.session_state.current_df, new_row], ignore_index=True, sort=False)
                persist_manual_rows(st.session_state.current_df)
                ev = evaluate_patient(ENGINE, new_row.iloc[0], None, None)
                st.success(f"{new_id.strip()} added · recommended Level {ev.recommended_priority} · {ev.uncertainty} uncertainty · {ev.current_risk} current risk.")
                st.caption(ev.next_best_information)

# ====================== REPEAT VITALS ======================
with repeat_tab:
    st.markdown("### Record Repeat Vitals")
    st.caption("This creates a real second snapshot for the selected live patient. PatientTriage.ai re-runs the same ML model and calculates ΔVitals/Δt plus the ML urgent-risk delta.")
    if queue.empty:
        st.info("No active patients.")
    else:
        pid = st.selectbox("Patient", queue["demo_id"].astype(str).tolist(), key="repeat_pid")
        row = patient_row(pid, active)
        current_elapsed = int(max(1, float(row.get("last_assessed_minutes", 0) or 0)))
        with st.form("repeat_vitals_form"):
            e0 = st.number_input("Minutes since previous measurement", min_value=1, max_value=240, value=max(10, current_elapsed), step=1)
            r1, r2, r3, r4 = st.columns(4)
            hr2 = float(r1.number_input("Heart rate", min_value=20.0, max_value=240.0, value=float(row.get("heart_rate", 85) if pd.notna(row.get("heart_rate")) else 85), step=1.0))
            rr2 = float(r2.number_input("Respiratory rate", min_value=4.0, max_value=60.0, value=float(row.get("respiratory_rate", 18) if pd.notna(row.get("respiratory_rate")) else 18), step=1.0))
            sp2 = float(r3.number_input("SpO₂ %", min_value=50.0, max_value=100.0, value=float(row.get("spo2", 98) if pd.notna(row.get("spo2")) else 98), step=0.1))
            temp2 = float(r4.number_input("Temperature °C", min_value=30.0, max_value=43.0, value=float(row.get("temperature_c", 37) if pd.notna(row.get("temperature_c")) else 37), step=0.1))
            bp2_available = st.checkbox("Blood pressure obtained", value=pd.notna(row.get("systolic_bp")))
            r5, r6, r7 = st.columns(3)
            sbp2 = float(r5.number_input("Systolic BP", min_value=40.0, max_value=260.0, value=float(row.get("systolic_bp", 125) if pd.notna(row.get("systolic_bp")) else 125), disabled=not bp2_available)) if bp2_available else np.nan
            dbp2 = float(r6.number_input("Diastolic BP", min_value=20.0, max_value=180.0, value=float(row.get("diastolic_bp", 75) if pd.notna(row.get("diastolic_bp")) else 75), disabled=not bp2_available)) if bp2_available else np.nan
            signal2 = r7.selectbox("SpO₂ signal quality", ["good", "noisy", "poor", "unreliable"], index=0 if str(row.get("spo2_signal_quality", "good")) == "good" else 1)
            repeat_submit = st.form_submit_button("Save repeat observations & recalculate", type="primary", use_container_width=True)
        if repeat_submit:
            current_df = st.session_state.current_df.copy()
            mask = current_df["demo_id"].astype(str).eq(pid)
            idx = current_df.index[mask][0]
            old = current_df.loc[idx].copy()
            st.session_state.previous_snapshots[pid] = old.to_dict()
            st.session_state.elapsed_since_prev[pid] = float(e0)
            updates = {
                "heart_rate": hr2, "respiratory_rate": rr2, "spo2": sp2, "temperature_c": temp2,
                "systolic_bp": sbp2, "diastolic_bp": dbp2, "spo2_signal_quality": signal2,
                "last_assessed_minutes": 0,
            }
            for k, v in updates.items():
                current_df.at[idx, k] = v
            st.session_state.current_df = current_df
            persist_manual_rows(current_df)
            new_row = current_df.loc[idx]
            ev = evaluate_patient(ENGINE, new_row, old, float(e0))
            st.success(f"Reassessment saved · Level {ev.recommended_priority} · {ev.trajectory} · Δ urgent risk {ev.ml_risk_delta*100:+.1f} pp")
            if ev.alert != "No active alert":
                st.warning(ev.alert)

# ====================== ACTIONS / AUDIT ======================
with action_tab:
    st.markdown("### Clinician Actions & Audit")
    if queue.empty:
        st.info("No active patients.")
    else:
        pid = st.selectbox("Patient", queue["demo_id"].astype(str).tolist(), key="action_pid")
        row = patient_row(pid, active)
        prev_dict = st.session_state.previous_snapshots.get(pid)
        prev = pd.Series(prev_dict) if prev_dict else None
        ev = evaluate_patient(ENGINE, row, prev, st.session_state.elapsed_since_prev.get(pid))
        st.write(f"AI recommendation: **Level {ev.recommended_priority}** · {ev.current_risk} risk · {ev.uncertainty} uncertainty")
        a1, a2 = st.columns([1, 1])
        with a1:
            final_level = st.selectbox("Clinician final level", [1, 2, 3, 4, 5], index=ev.recommended_priority - 1)
            reason = st.selectbox("Reason", [
                "Clinical assessment confirms recommendation",
                "Measurement artifact / repeat vital differs",
                "Additional history available",
                "Clinical appearance differs from model",
                "Protocol-specific judgement",
                "Other",
            ])
            note = st.text_input("Optional note")
            if st.button("Record clinician decision", type="primary", use_container_width=True):
                st.session_state.audit_log.append({
                    "sim_minute": st.session_state.sim_minute,
                    "patient": pid,
                    "action": "Clinician triage decision",
                    "ai_level": ev.recommended_priority,
                    "clinician_level": int(final_level),
                    "changed": int(final_level) != ev.recommended_priority,
                    "confidence": round(ev.confidence, 2),
                    "urgent_probability": round(ev.urgent_probability, 4),
                    "reason": reason,
                    "note": note,
                    "model_version": MODEL_VERSION,
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                })
                st.success("Decision recorded.")
        with a2:
            st.markdown("#### Remove / disposition")
            remove_reason = st.selectbox("Removal reason", ["Moved to treatment area", "Discharged from waiting list", "Transferred", "Left before assessment", "Demo cleanup", "Other"])
            st.caption("Removing a demo patient from the live list is session-local. Deleting its row from demo_patients.csv removes it from the dashboard source itself.")
            if st.button("Remove patient from live list", use_container_width=True):
                remove_from_live_list(pid, remove_reason)
                st.success(f"{pid} removed from the live list.")
                st.rerun()

        st.markdown("#### Audit trail")
        if st.session_state.audit_log:
            st.dataframe(pd.DataFrame(st.session_state.audit_log), use_container_width=True, hide_index=True)
        else:
            st.info("No clinician actions logged in this session yet.")

# ====================== TECHNICAL VALIDATION ======================
with validation_tab:
    st.markdown("### Technical Validation")
    st.caption("Separated from the clinical board so ED users see operational information first. These metrics are for technical review of the synthetic benchmark only.")
    metrics = json.load(open(METRICS_JSON))
    locked = metrics["locked_test"]
    arg = locked["argmax"]
    safe = locked["safety_operating_point"]
    a, b, c, d = st.columns(4)
    a.metric("Locked-test QWK", f"{safe['qwk']:.3f}")
    b.metric("Undertriage proxy", f"{safe['undertriage_rate']*100:.2f}%", f"vs {arg['undertriage_rate']*100:.2f}% argmax")
    c.metric("Severe undertriage", f"{safe['severe_undertriage_rate']*100:.3f}%")
    d.metric("ESI-1 recall", f"{safe['recall_esi1']*100:.2f}%")

    t1, t2 = st.columns(2)
    with t1:
        st.markdown("#### Safety trade-off")
        trade = pd.DataFrame({
            "Metric": ["Accuracy", "Macro F1", "Undertriage", "Overtriage", "ESI-1 recall", "ESI-2 recall"],
            "Plain argmax": [arg['accuracy'], arg['macro_f1'], arg['undertriage_rate'], arg['overtriage_rate'], arg['recall_esi1'], arg['recall_esi2']],
            "Safety-biased": [safe['accuracy'], safe['macro_f1'], safe['undertriage_rate'], safe['overtriage_rate'], safe['recall_esi1'], safe['recall_esi2']],
        })
        for cc in ["Plain argmax", "Safety-biased"]:
            trade[cc] = (100 * trade[cc]).round(2).astype(str) + "%"
        st.dataframe(trade, use_container_width=True, hide_index=True)
    with t2:
        st.markdown("#### Model evaluation protocol")
        st.write("• 68,000-record development pool; 12,000-record locked holdout.")
        st.write("• 5-fold stratified development cross-validation.")
        st.write("• Site-held-out generalization within development data.")
        st.write("• Deployment model excludes the locked holdout from fitting.")
        st.write("• Post-triage leakage and raw synthetic complaint-text shortcut are excluded.")

    cv_path = ROOT / "final_evaluation" / "five_fold_per_fold_metrics.csv"
    site_path = ROOT / "final_evaluation" / "site_held_out_evaluation_development_only.csv"
    if cv_path.exists():
        with st.expander("5-fold details"):
            st.dataframe(pd.read_csv(cv_path), use_container_width=True, hide_index=True)
    if site_path.exists():
        with st.expander("Site-held-out details"):
            st.dataframe(pd.read_csv(site_path), use_container_width=True, hide_index=True)
    if IMPORTANCE_CSV.exists():
        with st.expander("Top model features"):
            st.dataframe(pd.read_csv(IMPORTANCE_CSV).head(20), use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("PatientTriage.ai is a competition proof-of-concept using synthetic Triagegeist data. It is not clinically validated, not a medical device, and must not be used for patient care.")

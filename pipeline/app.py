

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from overtime_pipeline.hours import BREACH_THRESHOLD_HOURS, compute_all_weekly_hours
from overtime_pipeline.features import build_current_snapshot, build_training_table
from overtime_pipeline.model import fit_final_model, model_predict, walk_forward_evaluate
from overtime_pipeline.validate import validate_against_weekly_summary

st.set_page_config(page_title="Overtime Watch", layout="centered")

REQUIRED_FILES = [
    "shifts.csv", "employees.csv", "sites.csv", "public_holidays.csv",
    "weekly_summary.csv", "shift_notes.csv", "payroll_details.csv",
]
FILE_DESCRIPTIONS = {
    "shifts.csv": "Clock-in / clock-out records for the week",
    "employees.csv": "The full list of employees",
    "sites.csv": "The list of sites",
    "public_holidays.csv": "Public holiday dates",
    "weekly_summary.csv": "The client's own weekly totals - used to double-check our numbers are right",
    "shift_notes.csv": "Notes supervisors typed against shifts",
    "payroll_details.csv": "Payroll details (not used in the predictions)",
}
BUNDLED_DATA_DIR = Path(__file__).parent.parent / "data"


@st.cache_data(show_spinner=False)
def _run_pipeline(data_dir: str):
    employees = pd.read_csv(f"{data_dir}/employees.csv")
    sites = pd.read_csv(f"{data_dir}/sites.csv")

    weekly_hours, shifts, data_cutoff, current_week_start = compute_all_weekly_hours(data_dir)
    validation = validate_against_weekly_summary(weekly_hours, data_dir)

    elapsed_days = (data_cutoff - current_week_start).days + 1
    completed_weeks = sorted(w for w in weekly_hours["week_starting"].unique() if w < current_week_start)

    table = build_training_table(weekly_hours, shifts, employees, completed_weeks, elapsed_days)
    snapshot = build_current_snapshot(weekly_hours, employees, completed_weeks, current_week_start)

    wf = walk_forward_evaluate(table, elapsed_days) if len(completed_weeks) >= 2 else None
    clf = fit_final_model(table)
    preds = model_predict(clf, snapshot)

    preds = employees[["employee_id"]].merge(preds, on="employee_id", how="left")
    preds["will_breach"] = preds["will_breach"].fillna(0).astype(int)
    preds["risk_score"] = preds["risk_score"].fillna(0.0)

    current_hours = weekly_hours[weekly_hours["week_starting"] == current_week_start][
        ["employee_id", "total_hours"]
    ]
    remaining_days = max(7 - elapsed_days, 0)

    detail = preds.merge(current_hours, on="employee_id", how="left")
    detail = detail.merge(employees[["employee_id", "full_name", "role", "primary_site_id"]], on="employee_id", how="left")
    detail = detail.merge(sites, left_on="primary_site_id", right_on="site_id", how="left")
    detail = detail.merge(snapshot[["employee_id", "breach_rate_hist"]], on="employee_id", how="left")

    detail["headroom_hours"] = (BREACH_THRESHOLD_HOURS - detail["total_hours"]).round(2)
    detail["days_remaining"] = remaining_days
    detail["max_hours_per_remaining_day"] = (
        detail["headroom_hours"] / remaining_days if remaining_days > 0 else detail["headroom_hours"]
    ).round(1)

    return {
        "detail": detail, "validation": validation, "wf": wf,
        "sites": sites, "data_cutoff": data_cutoff,
        "current_week_start": current_week_start, "elapsed_days": elapsed_days,
    }


def _action_for(row) -> str:
    if row["headroom_hours"] <= 0:
        return "Already over the cap - pull them off remaining shifts this week."
    if row["days_remaining"] <= 0:
        return "Week is over - review Monday during payroll."
    if row["max_hours_per_remaining_day"] < 8:
        return (f"Cap remaining shifts to about {row['max_hours_per_remaining_day']:.1f}h/day over the next "
                f"{int(row['days_remaining'])} day(s), or pull one shift, to stay under the cap.")
    # headroom looks fine by simple math, but the model still flagged them -
    # that's their HISTORY driving the risk, not this week's pace yet, so say so
    hist_pct = row["breach_rate_hist"] * 100 if pd.notna(row["breach_rate_hist"]) else 0
    if hist_pct > 0:
        return (f"Flagged mainly on history - breached {hist_pct:.0f}% of past weeks. "
                f"On pace for now, but worth a check-in before the weekend.")
    return f"Model flags {row['risk_score']:.0%} risk despite {row['headroom_hours']:.1f}h headroom - worth a quick check-in."


def _compliance_gauge(score: float, label: str):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "", "font": {"size": 40}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "rgba(0,0,0,0)"},
            "steps": [
                {"range": [0, 50], "color": "#e74c3c"},
                {"range": [50, 80], "color": "#f39c12"},
                {"range": [80, 100], "color": "#2ecc71"},
            ],
            "threshold": {"line": {"color": "black", "width": 4}, "thickness": 0.9, "value": score},
        },
        title={"text": label},
    ))
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def main():
    st.title("Overtime watch")
    st.caption("Who's going to breach the 10-hour overtime cap by Sunday, and what to do about it today.")

    with st.sidebar:
        st.header("Load data")
        st.caption("Upload each file below to check a new week. Skip any file and last week's version is used instead.")
        uploaded_files = {}
        for req in REQUIRED_FILES:
            uploaded_files[req] = st.file_uploader(
                f"{req}", type="csv", key=f"upload_{req}",
                help=FILE_DESCRIPTIONS[req],
            )
            st.caption(FILE_DESCRIPTIONS[req])
        run_clicked = st.button("Run / refresh", type="primary", use_container_width=True)
        reset_clicked = st.button("Reset to sample data", use_container_width=True)

    if "data_dir" not in st.session_state:
        st.session_state.data_dir = str(BUNDLED_DATA_DIR)

    if reset_clicked:
        st.session_state.data_dir = str(BUNDLED_DATA_DIR)
        _run_pipeline.clear()
    elif run_clicked:
        any_uploaded = any(uploaded_files[req] is not None for req in REQUIRED_FILES)
        if any_uploaded:
            tmp_dir = tempfile.mkdtemp(prefix="overtime_data_")
            for req in REQUIRED_FILES:
                f = uploaded_files[req]
                if f is not None:
                    with open(Path(tmp_dir) / req, "wb") as out:
                        out.write(f.getbuffer())
                else:
                    shutil.copy(BUNDLED_DATA_DIR / req, Path(tmp_dir) / req)
            st.session_state.data_dir = tmp_dir
        else:
            st.session_state.data_dir = str(BUNDLED_DATA_DIR)
        _run_pipeline.clear()

    with st.spinner("Computing hours, validating, training..."):
        result = _run_pipeline(st.session_state.data_dir)

    v = result["validation"]
    if v["clean_match"]:
        st.success(f"Ground-truth check passed - {v['matched_rows']}/{v['client_rows']} rows match exactly.")
    else:
        st.warning(f"{v['total_hours_mismatches']} hour mismatches, {v['breached_mismatches']} breach mismatches found - review before trusting this export.")

    st.caption(
        f"Data through {result['data_cutoff'].date()} - in-progress week starting "
        f"{result['current_week_start'].date()} ({result['elapsed_days']}/7 days elapsed)."
    )

    detail = result["detail"]
    sites = result["sites"]

    # --- Site filter drives the gauge (name search is just a lookup, doesn't change the metric) ---
    site_options = ["All sites"] + sorted(sites["site_name"].dropna().unique().tolist())
    site_choice = st.selectbox("Filter by site", site_options)

    site_scoped = detail if site_choice == "All sites" else detail[detail["site_name"] == site_choice]
    compliance_score = round(100 * (site_scoped["will_breach"] == 0).mean(), 1)
    gauge_label = "Compliance score this week" if site_choice == "All sites" else f"Compliance score - {site_choice}"
    st.plotly_chart(_compliance_gauge(compliance_score, gauge_label), use_container_width=True)

    search = st.text_input("Search by name or employee ID", placeholder="Type a name or ID (e.g. E1042)...")

    view = site_scoped.copy()
    if search:
        mask = (
            view["full_name"].str.contains(search, case=False, na=False)
            | view["employee_id"].str.contains(search, case=False, na=False)
        )
        view = view[mask]

    flagged = view[view["will_breach"] == 1].sort_values("risk_score", ascending=False)
    others = view[view["will_breach"] == 0].sort_values("risk_score", ascending=False)

    st.subheader(f"{len(flagged)} of {len(view)} flagged")

    if flagged.empty:
        st.info("Nobody matching this search/filter is currently flagged.")
    else:
        for _, row in flagged.iterrows():
            with st.container(border=True):
                st.markdown(f"**{row['full_name']}**  \n{row['role']} - {row.get('site_name', row['primary_site_id'])}")
                st.metric("Chance of going over this week", f"{row['risk_score']*100:.0f}%", label_visibility="visible")
                st.caption(
                    f"Worked {row['total_hours']:.1f}h so far this week - "
                    f"{row['headroom_hours']:.1f}h left before the 55-hour legal limit."
                )
                st.markdown(f"**What to do:** {_action_for(row)}")

    with st.expander(f"Everyone else ({len(others)})"):
        st.dataframe(
            others[["employee_id", "full_name", "role", "site_name", "total_hours", "risk_score"]],
            use_container_width=True, hide_index=True,
        )

    if result["wf"] is not None:
        with st.expander("Model validation (walk-forward, baseline vs xgboost)"):
            pooled = result["wf"]["pooled"]
            st.markdown("**Naive baseline**")
            st.json(pooled["baseline"])
            st.markdown("**XGBoost**")
            st.json(pooled["xgboost"])
            st.caption("F2 is the headline metric - missing a breach costs more than a false alarm.")

    st.download_button(
        "Download predictions.csv",
        detail[["employee_id", "will_breach", "risk_score"]].to_csv(index=False),
        file_name="predictions.csv", mime="text/csv",
    )


if __name__ == "__main__":
    main()

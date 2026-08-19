"""
Cross-checks our computed hours against the client's own weekly_summary.csv.

"""
from __future__ import annotations

import pandas as pd


def validate_against_weekly_summary(weekly_hours: pd.DataFrame, data_dir: str) -> dict:
    ws = pd.read_csv(f"{data_dir}/weekly_summary.csv")
    ws["week_starting"] = pd.to_datetime(ws["week_starting"])

    merged = ws.merge(
        weekly_hours[["employee_id", "week_starting", "total_hours", "breached"]],
        on=["employee_id", "week_starting"], how="left", suffixes=("_client", "_ours"),
    )

    hours_diff = (merged["total_hours_client"] - merged["total_hours_ours"]).abs()
    hour_mismatches = merged[hours_diff > 0.01]

    breach_mismatch_mask = merged["breached_client"].astype(bool) != merged["breached_ours"].astype(bool)
    breach_mismatches = merged[breach_mismatch_mask]

    clean = len(hour_mismatches) == 0 and len(breach_mismatches) == 0

    return {
        "client_rows": len(ws),
        "matched_rows": len(merged),
        "total_hours_mismatches": len(hour_mismatches),
        "breached_mismatches": len(breach_mismatches),
        "clean_match": bool(clean),
        "mismatch_examples": hour_mismatches.head(5)[
            ["employee_id", "week_starting", "total_hours_client", "total_hours_ours"]
        ].astype(str).to_dict("records"),
    }

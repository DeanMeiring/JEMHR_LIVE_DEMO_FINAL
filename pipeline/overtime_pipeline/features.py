"""
Feature engineering - leakage-safe by construction.

Four features, exactly as reasoned out:
1. avg_hours_hist    - average weekly hours across all PRIOR completed weeks
2. std_hours_hist    - spread (std dev) of those prior weekly hours
3. breach_rate_hist  - fraction of those prior weeks that breached
4. current_pace_hours - hours worked Mon-Wed (or whatever elapsed_days is)
                         of the week being predicted, and nothing later

The label (breached) is the FULL week's outcome - only ever used for grading,
never as an input.

The critical rule: for a historical training week, features 1-3 only ever
look at weeks strictly BEFORE it, and feature 4 only ever looks at the first
`elapsed_days` days of THAT SAME week - exactly matching what would really
be known on a Wednesday. No week's Thu-Sun data ever leaks into its own
features or label being used as an input.
"""
from __future__ import annotations

import pandas as pd


def _current_pace_from_shifts(shifts: pd.DataFrame, week_start, elapsed_days: int) -> pd.DataFrame:
    """Sum each employee's shift hours for the first `elapsed_days` days of week_start only."""
    cutoff_date = week_start + pd.Timedelta(days=elapsed_days - 1)
    window = shifts[(shifts["week_starting"] == week_start) & (shifts["shift_date"] <= cutoff_date)]
    return (
        window.groupby("employee_id")["duration_hours"]
        .sum()
        .reset_index()
        .rename(columns={"duration_hours": "current_pace_hours"})
    )


def _history_features(weekly_hours: pd.DataFrame, prior_weeks: list) -> pd.DataFrame:
    """avg/std/breach_rate computed ONLY from weeks strictly before the target week."""
    hist = weekly_hours[weekly_hours["week_starting"].isin(prior_weeks)]
    agg = hist.groupby("employee_id").agg(
        avg_hours_hist=("total_hours", "mean"),
        std_hours_hist=("total_hours", "std"),
        breach_rate_hist=("breached", "mean"),
    ).reset_index()
    agg["std_hours_hist"] = agg["std_hours_hist"].fillna(0.0)  # a single prior week has no spread
    return agg


def build_training_table(weekly_hours: pd.DataFrame, shifts: pd.DataFrame,
                          employees: pd.DataFrame, completed_weeks: list, elapsed_days: int) -> pd.DataFrame:
    """
    One row per (employee_id, week) for every completed week that has at
    least one PRIOR completed week to build history from (the very first
    week in the dataset can't be used as a training example - no history
    exists yet - but it still counts as history for the second week).
    """
    rows = []
    for i, week in enumerate(completed_weeks):
        prior_weeks = completed_weeks[:i]
        if not prior_weeks:
            continue  # no history available yet - skip as a training row

        hist = _history_features(weekly_hours, prior_weeks)
        pace = _current_pace_from_shifts(shifts, week, elapsed_days)
        label = weekly_hours[weekly_hours["week_starting"] == week][["employee_id", "breached"]]

        merged = employees[["employee_id"]].merge(hist, on="employee_id", how="left")
        merged = merged.merge(pace, on="employee_id", how="left")
        merged = merged.merge(label, on="employee_id", how="left")
        merged["current_pace_hours"] = merged["current_pace_hours"].fillna(0.0)
        merged["week_starting"] = week
        rows.append(merged)

    table = pd.concat(rows, ignore_index=True)
    table = table.dropna(subset=["avg_hours_hist"])  # extra safety - drop anyone still missing history
    return table


def build_current_snapshot(weekly_hours: pd.DataFrame, employees: pd.DataFrame,
                            completed_weeks: list, current_week_start) -> pd.DataFrame:
    """
    Features for the LIVE in-progress week - history comes from every
    completed week we have, current pace is just that week's total_hours
    so far (weekly_hours already stops exactly at the real data cutoff).
    """
    hist = _history_features(weekly_hours, completed_weeks)
    current = weekly_hours[weekly_hours["week_starting"] == current_week_start][["employee_id", "total_hours"]]
    current = current.rename(columns={"total_hours": "total_hours"})
    current = current.rename(columns={"total_hours": "current_pace_hours"})

    snapshot = employees[["employee_id"]].merge(hist, on="employee_id", how="left")
    snapshot = snapshot.merge(current, on="employee_id", how="left")
    snapshot["current_pace_hours"] = snapshot["current_pace_hours"].fillna(0.0)
    return snapshot
"""
Weekly ordinary/overtime hours per employee.

Built from five ideas:

1. A shift's clock_out can be earlier than its clock_in on the clock face -
   that means it crossed midnight, so add 24h to clock_out before subtracting.
2. shift_date is fixed to a single calendar day (an overnight shift is never
   split across two dates), so each shift belongs to exactly one week.
3. Group shifts by employee_id + week_starting (Monday) and sum hours.
4. Left-join that onto the FULL employees.csv list, so an employee with zero
   shifts this week still gets a row (0 hours), instead of vanishing.
5. Ordinary cap = 45h. Overtime = hours above 45. Breach = total hours > 55
   (45 ordinary + 10 overtime, the BCEA hard cap from the README).
"""
from __future__ import annotations

import pandas as pd

ORDINARY_CAP_HOURS = 45
OVERTIME_CAP_HOURS = 10
BREACH_THRESHOLD_HOURS = ORDINARY_CAP_HOURS + OVERTIME_CAP_HOURS  # 55


def _shift_duration_hours(clock_in: str, clock_out: str):
    """Step 1: subtract clock_in from clock_out, fixing midnight crossings."""
    if not clock_in or not clock_out:
        return None  # a blank clock_out - shift never got closed out, contributes 0
    fmt = "%H:%M"
    t_in = pd.to_datetime(clock_in, format=fmt)
    t_out = pd.to_datetime(clock_out, format=fmt)
    if t_out < t_in:
        t_out += pd.Timedelta(days=1)  # crossed midnight - add 24h
    return (t_out - t_in).total_seconds() / 3600


def _week_starting(shift_date: pd.Series) -> pd.Series:
    """Step 2/3: Monday of the week each shift_date falls in."""
    return shift_date - pd.to_timedelta(shift_date.dt.weekday, unit="D")


def compute_all_weekly_hours(data_dir: str):
    """
    Returns:
        weekly_hours: one row per (employee_id, week_starting), every employee
                      present every week even with 0 shifts.
        shifts: the raw shifts table with duration_hours attached.
        data_cutoff: the latest shift_date in the data (our "today").
        current_week_start: Monday of the in-progress week.
    """
    shifts = pd.read_csv(f"{data_dir}/shifts.csv", dtype=str)
    employees = pd.read_csv(f"{data_dir}/employees.csv")

    shifts["shift_date"] = pd.to_datetime(shifts["shift_date"])
    shifts["duration_hours"] = shifts.apply(
        lambda r: _shift_duration_hours(r["clock_in_time"], r["clock_out_time"]), axis=1
    ).fillna(0.0)
    shifts["week_starting"] = _week_starting(shifts["shift_date"])

    # Step 3: group + sum
    weekly = (
        shifts.groupby(["employee_id", "week_starting"])["duration_hours"]
        .sum()
        .reset_index()
        .rename(columns={"duration_hours": "total_hours"})
    )

    # Step 4: left-join onto the FULL employee x week grid, so nobody vanishes
    all_weeks = sorted(shifts["week_starting"].unique())
    grid = pd.MultiIndex.from_product(
        [employees["employee_id"], all_weeks], names=["employee_id", "week_starting"]
    ).to_frame(index=False)
    weekly_hours = grid.merge(weekly, on=["employee_id", "week_starting"], how="left")
    weekly_hours["total_hours"] = weekly_hours["total_hours"].fillna(0.0)

    # Step 5: overtime + breach
    weekly_hours["overtime_hours"] = (weekly_hours["total_hours"] - ORDINARY_CAP_HOURS).clip(lower=0)
    weekly_hours["breached"] = weekly_hours["overtime_hours"] > OVERTIME_CAP_HOURS

    weekly_hours = weekly_hours.merge(employees, on="employee_id", how="left")

    data_cutoff = shifts["shift_date"].max()
    current_week_start = _week_starting(pd.Series([data_cutoff]))[0]

    return weekly_hours, shifts, data_cutoff, current_week_start
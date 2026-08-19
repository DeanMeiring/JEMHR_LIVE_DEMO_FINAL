"""
Runs the whole pipeline end to end:
  1. Compute weekly hours from raw shifts (hours.py)
  2. Build leakage-safe features for training + the live snapshot (features.py)
  3. Walk-forward evaluate baseline vs XGBoost, honestly (model.py)
  4. Fit the final model on all available history
  5. Predict the in-progress week and write predictions.csv

Usage: python run.py --data-dir ../data --out-dir .
Rerunning against a new data/ folder requires no code changes - the week
boundaries and elapsed_days are derived from the data itself, not hardcoded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from overtime_pipeline.validate import validate_against_weekly_summary
import pandas as pd

from overtime_pipeline.hours import compute_all_weekly_hours
from overtime_pipeline.features import build_training_table, build_current_snapshot
from overtime_pipeline.model import walk_forward_evaluate, fit_final_model, model_predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../data")
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    employees = pd.read_csv(f"{args.data_dir}/employees.csv")

    print("Computing weekly hours...")
    weekly_hours, shifts, data_cutoff, current_week_start = compute_all_weekly_hours(args.data_dir)
    elapsed_days = (data_cutoff - current_week_start).days + 1
    print(f"  data through {data_cutoff.date()} | in-progress week starting {current_week_start.date()} ({elapsed_days}/7 days elapsed)")
    weekly_hours.to_csv(out_dir / "weekly_hours.csv", index=False)

    completed_weeks = sorted(w for w in weekly_hours["week_starting"].unique() if w < current_week_start)
    print(f"  {len(completed_weeks)} completed weeks available")

    print("Validating against the client's own weekly_summary.csv...")
    validation = validate_against_weekly_summary(weekly_hours, args.data_dir)
    with open(out_dir / "validation.json", "w") as fjson:
        json.dump(validation, fjson, indent=2, default=str)
    if validation["clean_match"]:
        print(f"  PASS - {validation['matched_rows']}/{validation['client_rows']} rows match exactly, 0 mismatches")
    else:
        print(f"  WARNING - {validation['total_hours_mismatches']} hour mismatches, "
              f"{validation['breached_mismatches']} breach mismatches out of {validation['matched_rows']} rows")
        for ex in validation["mismatch_examples"]:
            print(f"    {ex}")

    print("Building features...")
    table = build_training_table(weekly_hours, shifts, employees, completed_weeks, elapsed_days)
    snapshot = build_current_snapshot(weekly_hours, employees, completed_weeks, current_week_start)
    print(f"  {len(table)} training rows | {len(snapshot)} employees in live snapshot")

    print("Walk-forward evaluating baseline vs xgboost...")
    result = walk_forward_evaluate(table, elapsed_days)
    for f in result["folds"]:
        print(f"  {f['test_week']}: baseline F2={f['baseline']['f2']} | xgboost F2={f['xgboost']['f2']}")
    print("  POOLED:", result["pooled"])
    with open(out_dir / "walk_forward_metrics.json", "w") as fjson:
        json.dump(result, fjson, indent=2, default=str)

    print("Fitting final model on all available history...")
    clf = fit_final_model(table)

    print("Predicting the in-progress week...")
    preds = model_predict(clf, snapshot)

    # every employee_id in employees.csv must get a row, even if something upstream dropped them
    preds = employees[["employee_id"]].merge(preds, on="employee_id", how="left")
    preds["will_breach"] = preds["will_breach"].fillna(0).astype(int)
    preds["risk_score"] = preds["risk_score"].fillna(0.0)

    preds = preds[["employee_id", "will_breach", "risk_score"]]
    preds.to_csv(out_dir / "predictions.csv", index=False)
    print(f"Wrote predictions.csv - {preds['will_breach'].sum()} / {len(preds)} employees flagged")


if __name__ == "__main__":
    main()
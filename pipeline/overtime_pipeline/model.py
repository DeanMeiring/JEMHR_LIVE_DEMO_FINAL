"""
Walk-forward validation + XGBoost, compared honestly against the baseline.


"""
from __future__ import annotations

import math

import pandas as pd
from xgboost import XGBClassifier

from .hours import BREACH_THRESHOLD_HOURS

FEATURES = ["avg_hours_hist", "std_hours_hist", "breach_rate_hist", "current_pace_hours"]
SOFTEN = 10


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    z = math.exp(x)
    return z / (1 + z)


def _baseline_from_pace(current_pace_hours: pd.Series, elapsed_days: int) -> pd.Series:
    """Same formula as baseline.py, just fed from the training table's current_pace_hours column."""
    days_remaining = max(7 - elapsed_days, 0)
    daily_avg = current_pace_hours / max(elapsed_days, 1)
    projected = current_pace_hours + daily_avg * days_remaining
    margin = projected - BREACH_THRESHOLD_HOURS
    return margin.apply(lambda m: _sigmoid(m / SOFTEN))


def _counts(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, fp, fn


def _metrics(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f2 = (5 * precision * recall) / (4 * precision + recall) if (precision + recall) else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3), "f2": round(f2, 3), "tp": tp, "fp": fp, "fn": fn}


def walk_forward_evaluate(table: pd.DataFrame, elapsed_days: int) -> dict:
    usable_weeks = sorted(table["week_starting"].unique())
    fold_results = []
    pooled = {"baseline": [0, 0, 0], "xgboost": [0, 0, 0]}  # running [tp, fp, fn]

    for i in range(1, len(usable_weeks)):
        train_weeks = usable_weeks[:i]
        test_week = usable_weeks[i]
        train_df = table[table["week_starting"].isin(train_weeks)]
        test_df = table[table["week_starting"] == test_week]

        y_train = train_df["breached"].astype(int)
        y_test = test_df["breached"].astype(int)

        # --- baseline ---
        base_risk = _baseline_from_pace(test_df["current_pace_hours"], elapsed_days)
        base_pred = (base_risk >= 0.5).astype(int)
        b_tp, b_fp, b_fn = _counts(y_test.values, base_pred.values)

        # --- xgboost ---
        # breaches are rare (~3.5% of rows) - tell XGBoost that missing one
        # costs roughly (majority count / minority count) times more than a
        # false alarm, computed ONLY from this fold's own training data
        pos = y_train.sum()
        neg = len(y_train) - pos
        spw = (neg / pos) if pos > 0 else 1.0
        clf = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                             eval_metric="logloss", random_state=42, scale_pos_weight=spw)
        clf.fit(train_df[FEATURES], y_train)
        xgb_risk = clf.predict_proba(test_df[FEATURES])[:, 1]
        xgb_pred = (xgb_risk >= 0.5).astype(int)
        x_tp, x_fp, x_fn = _counts(y_test.values, xgb_pred)

        fold_results.append({
            "test_week": str(test_week.date()),
            "n_train_weeks": i,
            "baseline": _metrics(b_tp, b_fp, b_fn),
            "xgboost": _metrics(x_tp, x_fp, x_fn),
        })

        pooled["baseline"][0] += b_tp; pooled["baseline"][1] += b_fp; pooled["baseline"][2] += b_fn
        pooled["xgboost"][0] += x_tp; pooled["xgboost"][1] += x_fp; pooled["xgboost"][2] += x_fn

    pooled_metrics = {name: _metrics(*counts) for name, counts in pooled.items()}
    return {"folds": fold_results, "pooled": pooled_metrics}


def fit_final_model(table: pd.DataFrame) -> XGBClassifier:
    """Train on ALL usable historical weeks - this is the model used for the live prediction."""
    y = table["breached"].astype(int)
    pos = y.sum()
    neg = len(y) - pos
    spw = (neg / pos) if pos > 0 else 1.0
    clf = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                         eval_metric="logloss", random_state=42, scale_pos_weight=spw)
    clf.fit(table[FEATURES], y)
    return clf


def model_predict(clf: XGBClassifier, snapshot: pd.DataFrame) -> pd.DataFrame:
    out = snapshot[["employee_id"]].copy()
    out["risk_score"] = clf.predict_proba(snapshot[FEATURES])[:, 1].round(4)
    out["will_breach"] = (out["risk_score"] >= 0.5).astype(int)
    return out

"""Temporal validation of the bike-traffic model.

The Spark notebooks use a random 80/20 split, which lets the model see hours
from the same days it is tested on. This script re-evaluates the approach with
a strict chronological split (train on 1-22 April, test on 23-30 April) and
compares it with a seasonal-naive baseline (same hour, one week earlier).

Usage: python src/temporal_validation.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "data_final_spark.csv"
SPLIT = pd.Timestamp("2022-04-23")
FEATURES = ["temperature", "precipitation", "hour", "day_of_week",
            "is_weekend", "is_rush_hour", "is_raining"]


def metrics(y_true, y_pred):
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def main():
    df = pd.read_csv(DATA, sep=";", parse_dates=["timestamp"]).sort_values("timestamp")
    df = df.set_index("timestamp")

    train, test = df[df.index < SPLIT], df[df.index >= SPLIT]

    # Seasonal-naive baseline: traffic at the same hour 7 days earlier.
    lag_week = df["nb_velos"].shift(freq="7D").reindex(test.index)
    mask = lag_week.notna()

    model = GradientBoostingRegressor(n_estimators=300, max_depth=4,
                                      learning_rate=0.05, random_state=42)
    model.fit(train[FEATURES], train["nb_velos"])
    pred = model.predict(test[FEATURES])

    rows = {
        "Seasonal naive (t-7d)": metrics(test["nb_velos"][mask], lag_week[mask]),
        "Gradient Boosting": metrics(test["nb_velos"], pred),
    }
    print(f"Train: {len(train)} h ({train.index.min():%d/%m} -> {train.index.max():%d/%m})")
    print(f"Test : {len(test)} h ({test.index.min():%d/%m} -> {test.index.max():%d/%m})\n")
    print(pd.DataFrame(rows).T.round(3).to_string())

    importances = pd.Series(model.feature_importances_, index=FEATURES)
    print("\nFeature importance:\n" + importances.sort_values(ascending=False).round(3).to_string())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(test.index, test["nb_velos"], color="#1c1b19", lw=1.6, label="Actual")
    ax.plot(test.index, pred, color="#0f766e", lw=1.6, ls="--", label="Predicted (GBT)")
    ax.set_title(f"Hold-out week, 23–29 April 2022 — R² = {rows['Gradient Boosting']['R2']:.2f}")
    ax.set_ylabel("Bikes per hour (all counters)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "holdout_week.png", dpi=130)


if __name__ == "__main__":
    main()

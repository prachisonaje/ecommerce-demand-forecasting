
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "sales_data.csv"
MODELS = BASE / "models"
OUTPUTS = BASE / "outputs"
MODELS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)

CATEGORICAL = ["Store ID","Product ID","Category","Region","Weather Condition","Seasonality"]
NUMERIC = [
    "Inventory Level","Price","Discount","Promotion","Competitor Pricing","Epidemic",
    "Year","Month","Day","Day_of_Week","Week_of_Year","Is_Weekend","Quarter"
]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "Demand"

def add_calendar_features(data):
    data = data.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    data["Year"] = data["Date"].dt.year
    data["Month"] = data["Date"].dt.month
    data["Day"] = data["Date"].dt.day
    data["Day_of_Week"] = data["Date"].dt.dayofweek
    data["Week_of_Year"] = data["Date"].dt.isocalendar().week.astype(int)
    data["Is_Weekend"] = (data["Day_of_Week"] >= 5).astype(int)
    data["Quarter"] = data["Date"].dt.quarter
    return data

def make_preprocessor():
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", "passthrough", NUMERIC)
    ])

def main():
    df = pd.read_csv(DATA)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df = add_calendar_features(df)

    # Strict chronological split
    dates = np.sort(df["Date"].unique())
    cutoff = dates[int(len(dates) * 0.80)]
    train = df[df["Date"] < cutoff]
    test = df[df["Date"] >= cutoff]

    x_train, y_train = train[FEATURES], train[TARGET]
    x_test, y_test = test[FEATURES], test[TARGET]

    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=180, max_depth=18, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=450, max_depth=7, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85,
            objective="reg:squarederror", random_state=42, n_jobs=-1
        )
    }

    rows = []
    baseline = np.full(len(y_test), y_train.mean())
    rows.append(["Training Mean Baseline",
                 mean_absolute_error(y_test, baseline),
                 mean_squared_error(y_test, baseline) ** 0.5])

    for name, estimator in models.items():
        pipe = Pipeline([
            ("preprocessor", make_preprocessor()),
            ("model", estimator)
        ])
        pipe.fit(x_train, y_train)
        pred = pipe.predict(x_test)
        rows.append([
            name,
            mean_absolute_error(y_test, pred),
            mean_squared_error(y_test, pred) ** 0.5
        ])
        joblib.dump(pipe, MODELS / (name.lower().replace(" ","_") + "_pipeline.joblib"))

    comparison = pd.DataFrame(rows, columns=["Model","MAE","RMSE"])
    comparison.to_csv(OUTPUTS / "model_comparison.csv", index=False)

    # Final production model is trained on all historical data
    final_pipe = Pipeline([
        ("preprocessor", make_preprocessor()),
        ("model", XGBRegressor(
            n_estimators=450, max_depth=7, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85,
            objective="reg:squarederror", random_state=42, n_jobs=-1
        ))
    ])
    final_pipe.fit(df[FEATURES], df[TARGET])
    joblib.dump(final_pipe, MODELS / "demand_xgboost_final.joblib")

    meta = {
        "categorical": CATEGORICAL,
        "numeric": NUMERIC,
        "features": FEATURES,
        "stores": sorted(df["Store ID"].unique().tolist()),
        "products": sorted(df["Product ID"].unique().tolist()),
        "weather": sorted(df["Weather Condition"].unique().tolist()),
        "seasonality": sorted(df["Seasonality"].unique().tolist()),
        "training_end": str(df["Date"].max().date()),
        "cutoff": str(pd.Timestamp(cutoff).date())
    }
    with open(MODELS / "metadata.json","w") as f:
        json.dump(meta, f, indent=2)

    print(comparison.round(3).to_string(index=False))
    print("\nFinal model saved to models/demand_xgboost_final.joblib")

if __name__ == "__main__":
    main()

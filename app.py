
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "sales_data.csv"
MODEL = BASE / "models" / "demand_xgboost_final.joblib"
META = BASE / "models" / "metadata.json"
METRICS = BASE / "models" / "metrics.json"

st.set_page_config(page_title="E-commerce Demand Forecasting", page_icon="📈", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv(DATA)
    df["Date"] = pd.to_datetime(df["Date"])
    return df

@st.cache_resource
def load_model():
    return joblib.load(MODEL)

@st.cache_data
def load_json(path):
    with open(path) as f:
        return json.load(f)

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

df = load_data()
model = load_model()
meta = load_json(META)
metrics = load_json(METRICS)

st.title("📈 E-commerce Demand Forecasting")
st.caption("Machine-learning demand prediction and inventory-risk analysis")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Demand Forecast", "Inventory Risk", "Model Performance"])

if page == "Dashboard":
    st.subheader("Business Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", f"{len(df):,}")
    c2.metric("Stores", df["Store ID"].nunique())
    c3.metric("Products", df["Product ID"].nunique())
    c4.metric("Average Demand", f"{df['Demand'].mean():.1f}")

    st.markdown("### Demand trend")
    daily = df.groupby("Date", as_index=False)["Demand"].mean()
    st.line_chart(daily.set_index("Date")["Demand"])

    left, right = st.columns(2)
    with left:
        st.markdown("### Demand by category")
        cat = df.groupby("Category")["Demand"].mean().sort_values(ascending=False)
        st.bar_chart(cat)
    with right:
        st.markdown("### Demand by region")
        reg = df.groupby("Region")["Demand"].mean().sort_values(ascending=False)
        st.bar_chart(reg)

    st.info("The model predicts Demand. Units Sold is not used as an input feature because it can be directly related to the demand outcome and could introduce leakage depending on when the prediction is made.")

elif page == "Demand Forecast":
    st.subheader("Generate a Future Demand Forecast")

    store = st.selectbox("Store", meta["stores"])
    product = st.selectbox("Product", meta["products"])

    subset = df[(df["Store ID"] == store) & (df["Product ID"] == product)].sort_values("Date")
    if subset.empty:
        st.warning("No historical record found for this store/product combination.")
        st.stop()

    category = subset["Category"].mode().iloc[0]
    region = subset["Region"].mode().iloc[0]

    st.write(f"**Category:** {category}  |  **Region:** {region}")

    latest = subset.iloc[-1]
    col1, col2, col3 = st.columns(3)
    with col1:
        price = st.number_input("Price", min_value=0.0, value=float(latest["Price"]), step=1.0)
    with col2:
        discount = st.slider("Discount (%)", 0, 100, int(latest["Discount"]))
    with col3:
        inventory = st.number_input("Current Inventory", min_value=0, value=int(latest["Inventory Level"]), step=10)

    col4, col5, col6 = st.columns(3)
    with col4:
        promotion = st.selectbox("Promotion", [0, 1], index=int(latest["Promotion"]))
    with col5:
        weather = st.selectbox("Weather Condition", meta["weather"], index=meta["weather"].index(latest["Weather Condition"]))
    with col6:
        competitor = st.number_input("Competitor Pricing", min_value=0.0, value=float(latest["Competitor Pricing"]), step=1.0)

    epidemic = st.selectbox("Epidemic indicator", [0, 1], index=int(latest["Epidemic"]))
    horizon = st.slider("Forecast horizon (days)", 1, 30, 7)

    # Map month to the most frequent historical seasonality
    season_by_month = df.groupby(df["Date"].dt.month)["Seasonality"].agg(lambda x: x.mode().iloc[0]).to_dict()

    last_date = df["Date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

    if st.button("Generate Forecast", type="primary"):
        rows = []
        for d in future_dates:
            rows.append({
                "Date": d,
                "Store ID": store,
                "Product ID": product,
                "Category": category,
                "Region": region,
                "Inventory Level": inventory,
                "Price": price,
                "Discount": discount,
                "Weather Condition": weather,
                "Promotion": promotion,
                "Competitor Pricing": competitor,
                "Seasonality": season_by_month.get(d.month, latest["Seasonality"]),
                "Epidemic": epidemic
            })
        future = add_calendar_features(pd.DataFrame(rows))
        feature_cols = meta["features"]
        preds = model.predict(future[feature_cols])
        preds = np.maximum(preds, 0)
        result = pd.DataFrame({"Date": future_dates, "Predicted Demand": np.round(preds, 0).astype(int)})
        st.session_state["forecast_result"] = result

    if "forecast_result" in st.session_state:
        result = st.session_state["forecast_result"]
        st.dataframe(result, use_container_width=True, hide_index=True)
        st.line_chart(result.set_index("Date")["Predicted Demand"])
        st.metric("Total forecast demand", f"{int(result['Predicted Demand'].sum()):,} units")

elif page == "Inventory Risk":
    st.subheader("Inventory Risk Analysis")
    store = st.selectbox("Store", meta["stores"], key="risk_store")
    product = st.selectbox("Product", meta["products"], key="risk_product")
    subset = df[(df["Store ID"] == store) & (df["Product ID"] == product)].sort_values("Date")
    latest = subset.iloc[-1]

    current_inventory = st.number_input(
        "Current inventory", min_value=0, value=int(latest["Inventory Level"]), step=10
    )
    horizon = st.slider("Planning horizon (days)", 1, 30, 7, key="risk_horizon")

    season_by_month = df.groupby(df["Date"].dt.month)["Seasonality"].agg(lambda x: x.mode().iloc[0]).to_dict()
    future_dates = pd.date_range(df["Date"].max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    category = subset["Category"].mode().iloc[0]
    region = subset["Region"].mode().iloc[0]

    rows = []
    for d in future_dates:
        rows.append({
            "Date": d, "Store ID": store, "Product ID": product,
            "Category": category, "Region": region,
            "Inventory Level": current_inventory,
            "Price": float(latest["Price"]),
            "Discount": int(latest["Discount"]),
            "Weather Condition": latest["Weather Condition"],
            "Promotion": int(latest["Promotion"]),
            "Competitor Pricing": float(latest["Competitor Pricing"]),
            "Seasonality": season_by_month.get(d.month, latest["Seasonality"]),
            "Epidemic": int(latest["Epidemic"])
        })
    future = add_calendar_features(pd.DataFrame(rows))
    pred = np.maximum(model.predict(future[meta["features"]]), 0)
    total_demand = int(np.ceil(pred.sum()))
    gap = total_demand - current_inventory

    c1, c2, c3 = st.columns(3)
    c1.metric("Current inventory", f"{current_inventory:,}")
    c2.metric(f"Forecast demand ({horizon} days)", f"{total_demand:,}")
    c3.metric("Potential shortage", f"{max(gap,0):,}")

    if gap > 0:
        st.error(f"Potential stockout: forecast demand exceeds current inventory by about {gap:,} units.")
    else:
        st.success(f"Inventory covers the forecast demand with about {abs(gap):,} units remaining.")

    st.caption("This is a demand-vs-inventory risk indicator, not a full inventory-optimization/order-quantity calculation.")

elif page == "Model Performance":
    st.subheader("Model Evaluation")
    st.write(f"Chronological validation: training through **{metrics['train_end']}**, testing from **{metrics['test_start']}** to **{metrics['test_end']}**.")
    comp = pd.read_csv(BASE / "outputs" / "model_comparison.csv")
    st.dataframe(comp.round(3), use_container_width=True, hide_index=True)

    best = comp.loc[comp["MAE"].idxmin()]
    st.success(f"Best validation MAE in this comparison: {best['Model']} ({best['MAE']:.2f} units).")
    st.markdown("### Metric interpretation")
    st.write("- **MAE:** average absolute prediction error in demand units.")
    st.write("- **RMSE:** penalizes larger errors more strongly.")
    st.write("- **MAPE/WAPE:** percentage-style error measures; WAPE is safer when demand can be zero.")

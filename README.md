# E-Commerce Demand Forecasting Using Machine Learning

## 1. Project Overview

This project builds a machine-learning system to predict future retail/e-commerce demand and use those forecasts to identify potential inventory shortages.

The project uses a publicly provided retail inventory dataset containing 76,000 records, 760 daily dates, 5 stores, 20 products, and a numerical `Demand` target.

### Business problem

Retailers need to estimate future demand so they can plan inventory and reduce the risk of:

- stockouts
- excess inventory
- missed sales
- inefficient replenishment

## 2. Objective

Predict `Demand` using historical retail information such as:

- Store and product
- Category and region
- Inventory level
- Price
- Discount
- Promotion
- Weather
- Competitor pricing
- Seasonality
- Epidemic indicator
- Calendar features derived from date

**Important:** The first version intentionally does not use manually engineered lag or rolling-average features. They can be tested later as an experiment.

## 3. Dataset

Columns:

- `Date`
- `Store ID`
- `Product ID`
- `Category`
- `Region`
- `Inventory Level`
- `Units Sold`
- `Units Ordered`
- `Price`
- `Discount`
- `Weather Condition`
- `Promotion`
- `Competitor Pricing`
- `Seasonality`
- `Epidemic`
- `Demand`

There are no missing values in the supplied dataset.

### Target

`Demand`

### Leakage consideration

`Units Sold` and `Units Ordered` are not used as model inputs in the first version. Depending on the exact operational timing, these fields can be downstream of demand or purchasing decisions and can introduce leakage or unrealistic future information.

## 4. Methodology

```text
Raw Data
   |
Data Cleaning
   |
Date Conversion
   |
Calendar Feature Engineering
   |
Chronological Train/Test Split
   |
Baseline + Random Forest + XGBoost
   |
MAE / RMSE / WAPE / MAPE
   |
Final XGBoost Model
   |
Streamlit Application
   |
Demand Forecast + Inventory Risk
```

### Calendar features

The model uses:

- Year
- Month
- Day
- Day of week
- Week of year
- Weekend indicator
- Quarter

### Why chronological splitting?

Random splitting is avoided because forecasting must respect time order. The model is trained on earlier dates and evaluated on later dates.

## 5. Model Results

Using an 80/20 chronological split:

| Model | MAE | RMSE |
|---|---:|---:|
| Training Mean Baseline | 35.43 | 44.37 |
| Random Forest | 14.30 | 20.49 |
| XGBoost | **10.87** | **15.45** |

The XGBoost model was selected for the production/demo application.

Additional holdout metrics:

- WAPE: ~10.84%
- MAPE on non-zero actual demand: ~16.00%

These are validation results from the supplied dataset and should be regenerated if the data or training configuration changes.

## 6. Inventory Risk Logic

The application compares:

`Forecast Demand` vs `Current Inventory`

If:

`Forecast Demand > Current Inventory`

the application flags a potential stockout.

This is **not** a complete inventory optimization system. A production ordering engine would additionally require supplier lead time, safety stock, holding cost, stockout cost, minimum order quantities, and other supply-chain constraints.

## 7. Streamlit Features

The app contains:

### Dashboard
- Total records
- Stores
- Products
- Average demand
- Demand trend
- Demand by category
- Demand by region

### Demand Forecast
- Store selection
- Product selection
- Price
- Discount
- Promotion
- Weather
- Competitor price
- Inventory
- Forecast horizon
- Forecast table and chart

### Inventory Risk
- Current inventory
- Forecast demand
- Potential shortage
- Stockout warning

### Model Performance
- Model comparison
- MAE
- RMSE
- Validation period

## 8. How to Run

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If you want to retrain everything:

```bash
python training.py
```

Run Streamlit:

```bash
streamlit run app.py
```

## 9. Project Structure

```text
ecommerce_demand_forecasting/
│
├── app.py
├── training.py
├── requirements.txt
├── README.md
│
├── data/
│   └── sales_data.csv
│
├── models/
│   ├── demand_xgboost_final.joblib
│   ├── demand_xgboost_pipeline.joblib
│   ├── random_forest_pipeline.joblib
│   ├── metrics.json
│   └── metadata.json
│
└── outputs/
    └── model_comparison.csv
```

## 10. Future Improvements

- Add lag and rolling-demand features and compare validation performance.
- Add walk-forward/rolling-origin validation.
- Test dedicated time-series models.
- Add prediction intervals.
- Add automated replenishment optimization.
- Incorporate holidays/events if available.
- Add model monitoring and drift detection.

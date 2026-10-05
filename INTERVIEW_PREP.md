# Interview Preparation — E-Commerce Demand Forecasting

## What is the project?
A machine-learning demand forecasting system. I predict the `Demand` target using retail, pricing, promotion, competition, environmental, seasonal, and calendar information, then compare the forecast with current inventory to flag potential stockout risk.

## Why did you choose it?
My existing credit-card fraud project is a classification problem. I wanted my second ML project to demonstrate a different problem type: demand forecasting with time-dependent data and a direct business application.

## Why is this time-series data?
Each observation has a date and observations occur over time. Demand can change according to weekly, monthly, seasonal, promotional, and other temporal patterns.

## Why not randomly split the data?
Forecasting must respect time order. I trained on earlier dates and evaluated on later dates so future information does not leak into training.

## What is the target?
`Demand`.

## Why not use Units Sold as an input?
Units Sold can be closely related to demand and may represent an outcome observed during the same period. Without a carefully defined prediction timestamp, it can cause leakage or make the model unrealistic for future forecasting. I excluded it from the first version.

## Did you use lag/rolling features?
Not in the first version. I started with calendar and business features to create a clear baseline. Lag and rolling features are optional experiments that can be added only if time-aware validation shows they improve performance.

## Why XGBoost?
XGBoost performs well on structured/tabular data and can model nonlinear relationships between demand and variables such as price, discount, promotion, inventory, seasonality, and competitor pricing.

## Is XGBoost a time-series model?
No. XGBoost is a gradient-boosted tree model. I apply it to time-dependent data using calendar features and a chronological validation strategy.

## What metrics did you use?
MAE and RMSE are primary metrics. I also calculated WAPE and non-zero MAPE. MAE is easy to interpret in demand units; RMSE penalizes larger errors more strongly.

## What were your results?
Chronological holdout:
- Mean baseline MAE: 35.43
- Random Forest MAE: 14.30
- XGBoost MAE: 10.87
- XGBoost RMSE: 15.45
- XGBoost WAPE: about 10.84%

## What is inventory risk?
I compare forecast demand over a selected planning horizon with current inventory. If forecast demand exceeds inventory, I flag potential stockout risk.

## Is this inventory optimization?
No. It is a demand-vs-inventory risk indicator. Full optimization would require safety stock, supplier lead time, ordering costs, holding costs, service levels, and other constraints.

## Why Streamlit?
It turns the trained model into an interactive application where a user can select a store/product, change assumptions, generate a forecast, and inspect inventory risk.

## Future improvements
I would test lag/rolling features, walk-forward validation, dedicated time-series models, prediction intervals, holiday/event data, and a more complete replenishment optimization layer.

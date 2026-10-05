
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"
df = pd.read_csv(BASE / "data" / "sales_data.csv")
df["Date"] = pd.to_datetime(df["Date"])

daily = df.groupby("Date", as_index=False)["Demand"].mean()
plt.figure(figsize=(12,5))
plt.plot(daily["Date"], daily["Demand"])
plt.title("Average Daily Demand")
plt.xlabel("Date")
plt.ylabel("Demand")
plt.tight_layout()
plt.savefig(OUT/"demand_over_time.png", dpi=160)
plt.close()

monthly = df.assign(Month=df["Date"].dt.month).groupby("Month")["Demand"].mean()
plt.figure(figsize=(9,5))
plt.bar(monthly.index.astype(str), monthly.values)
plt.title("Average Demand by Month")
plt.xlabel("Month")
plt.ylabel("Average Demand")
plt.tight_layout()
plt.savefig(OUT/"demand_by_month.png", dpi=160)
plt.close()

category = df.groupby("Category")["Demand"].mean().sort_values()
plt.figure(figsize=(9,5))
plt.barh(category.index, category.values)
plt.title("Average Demand by Category")
plt.xlabel("Average Demand")
plt.tight_layout()
plt.savefig(OUT/"demand_by_category.png", dpi=160)
plt.close()

print("EDA charts saved to outputs/")

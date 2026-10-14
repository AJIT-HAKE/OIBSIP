import pandas as pd
import os

df = pd.read_csv("outputs/cleaned_retail_data.csv")
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# Use the day after the last transaction as the reference date
reference_date = df["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)

rfm = df.groupby("CustomerID").agg(
    LastPurchase=("InvoiceDate", "max"),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("TotalAmount", "sum")
).reset_index()

rfm["Recency"] = (reference_date - rfm["LastPurchase"].dt.normalize()).dt.days
rfm = rfm[["CustomerID", "Recency", "Frequency", "Monetary"]]

os.makedirs("outputs", exist_ok=True)
rfm.to_csv("outputs/rfm_customer_data.csv", index=False)

print("REFERENCE DATE:", reference_date.date())
print("CUSTOMERS IN RFM:", len(rfm))
print("\nRFM SUMMARY:")
print(rfm[["Recency", "Frequency", "Monetary"]].describe().round(2).to_string())
print("\nTOP 10 CUSTOMERS BY MONETARY VALUE:")
print(rfm.sort_values("Monetary", ascending=False).head(10).to_string(index=False))
print("\nSaved: outputs/rfm_customer_data.csv")

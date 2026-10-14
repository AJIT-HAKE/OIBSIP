import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

os.makedirs("outputs", exist_ok=True)
sns.set_theme(style="whitegrid")

# Load customer-level RFM data
rfm = pd.read_csv("outputs/rfm_customer_data.csv")

features = ["Recency", "Frequency", "Monetary"]
X = rfm[features].copy()

# Log transformation reduces the impact of extreme values
X["Frequency"] = X["Frequency"].clip(lower=1)
X["Monetary"] = X["Monetary"].clip(lower=0)
X_log = X.copy()
X_log["Frequency"] = __import__("numpy").log1p(X_log["Frequency"])
X_log["Monetary"] = __import__("numpy").log1p(X_log["Monetary"])

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_log)

# Elbow method
inertias = []
k_values = range(2, 10)

for k in k_values:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X_scaled)
    inertias.append(model.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(list(k_values), inertias, marker="o")
plt.title("Elbow Method for Choosing K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.xticks(list(k_values))
plt.tight_layout()
plt.savefig("outputs/elbow_method.png", dpi=150)
plt.close()

# Use K=4 initially; we will review the elbow plot and profiles
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm["Cluster"] = kmeans.fit_predict(X_scaled)

# Save assigned customer segments
rfm.to_csv("outputs/customer_segments.csv", index=False)

# Cluster profiles in original RFM units
profiles = rfm.groupby("Cluster").agg(
    CustomerCount=("CustomerID", "count"),
    AvgRecencyDays=("Recency", "mean"),
    AvgFrequency=("Frequency", "mean"),
    AvgMonetary=("Monetary", "mean"),
    MedianMonetary=("Monetary", "median")
).round(2)

profiles.to_csv("outputs/cluster_profiles.csv")

# Bar chart: customer count per cluster
plt.figure(figsize=(8, 5))
sns.countplot(data=rfm, x="Cluster")
plt.title("Number of Customers in Each Cluster")
plt.xlabel("Cluster")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("outputs/cluster_customer_counts.png", dpi=150)
plt.close()

# Scatter plot 1: Recency vs Frequency
plt.figure(figsize=(8, 5))
sns.scatterplot(data=rfm, x="Recency", y="Frequency", hue="Cluster", palette="tab10", alpha=0.65)
plt.title("Customer Segments: Recency vs Frequency")
plt.tight_layout()
plt.savefig("outputs/recency_frequency_clusters.png", dpi=150)
plt.close()

# Scatter plot 2: Frequency vs Monetary
plt.figure(figsize=(8, 5))
sns.scatterplot(data=rfm, x="Frequency", y="Monetary", hue="Cluster", palette="tab10", alpha=0.65)
plt.title("Customer Segments: Frequency vs Monetary")
plt.tight_layout()
plt.savefig("outputs/frequency_monetary_clusters.png", dpi=150)
plt.close()

print("ELBOW INERTIAS:")
for k, inertia in zip(k_values, inertias):
    print(f"K={k}: {inertia:.2f}")

print("\nCLUSTER PROFILES:")
print(profiles.to_string())

print("\nSaved files:")
for name in [
    "elbow_method.png",
    "cluster_customer_counts.png",
    "recency_frequency_clusters.png",
    "frequency_monetary_clusters.png",
    "customer_segments.csv",
    "cluster_profiles.csv"
]:
    print("outputs/" + name)

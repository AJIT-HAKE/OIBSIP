import nbformat as nbf

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    }
}

nb["cells"] = [
    nbf.v4.new_markdown_cell("""# House Price Prediction
## Oasis Infobyte Data Analytics Internship — Level 2

**Objective:** Build and evaluate a Linear Regression model using the Boston Housing dataset.

**Target:** `medv` — median owner-occupied home value, expressed in thousands of US dollars in this dataset's original context.

**Important:** This is a historical Boston-area dataset, not an Indian property-price dataset."""),
    nbf.v4.new_markdown_cell("## 1. Import libraries and load data"),
    nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("data/train.csv")
print("Shape:", df.shape)
display(df.head())
display(df.describe().T)"""),
    nbf.v4.new_markdown_cell("## 2. Data quality and cleaning"),
    nbf.v4.new_code_cell("""print("Missing values before cleaning:", df.isna().sum().sum())
print("Duplicate rows before cleaning:", df.duplicated().sum())

df = df.drop_duplicates().copy()
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna()
df = df[df["medv"] > 0]

display(pd.DataFrame({
    "Data Type": df.dtypes.astype(str),
    "Missing Values": df.isna().sum(),
    "Unique Values": df.nunique()
}))
print("Shape after cleaning:", df.shape)"""),
    nbf.v4.new_markdown_cell("## 3. Exploratory Data Analysis"),
    nbf.v4.new_code_cell("""df.hist(figsize=(14, 10), bins=20)
plt.suptitle("Feature Distributions")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
plt.scatter(df["rm"], df["medv"], alpha=0.65)
plt.xlabel("Average Number of Rooms")
plt.ylabel("Median House Value (thousands of USD)")
plt.title("Rooms vs House Value")
plt.show()

display(df.corr(numeric_only=True)["medv"].sort_values(ascending=False))"""),
    nbf.v4.new_markdown_cell("## 4. Split data and train Linear Regression"),
    nbf.v4.new_code_cell("""X = df.drop(columns=["medv"])
y = df["medv"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))"""),
    nbf.v4.new_markdown_cell("## 5. Evaluate the model"),
    nbf.v4.new_code_cell("""mse = mean_squared_error(y_test, predictions)
metrics = pd.DataFrame([{
    "MAE": mean_absolute_error(y_test, predictions),
    "MSE": mse,
    "RMSE": np.sqrt(mse),
    "R2 Score": r2_score(y_test, predictions)
}])
display(metrics)

plt.figure(figsize=(7, 6))
plt.scatter(y_test, predictions, alpha=0.7)
low = min(y_test.min(), predictions.min())
high = max(y_test.max(), predictions.max())
plt.plot([low, high], [low, high], "--")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted House Values")
plt.show()

display(pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions.round(2)
}).head(10))"""),
    nbf.v4.new_markdown_cell("""## 6. Conclusion

The Linear Regression model was trained using an 80/20 train-test split. Its performance is evaluated with MAE, RMSE, and R².

**Limitations:** This is a historical dataset from Boston, not India. Results on this dataset should not be used to estimate current property prices. Model performance depends on the dataset and chosen features.""")
]

nbf.write(nb, "House_Price_Prediction.ipynb")
print("Notebook created successfully!")

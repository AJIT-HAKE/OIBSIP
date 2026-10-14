# House Price Prediction — Oasis Infobyte

## Level 2 Machine Learning Project

### Objective
Build a Linear Regression model to estimate median home values from housing-related features.

### Dataset
Boston Housing dataset, 506 rows and 14 columns, including `medv` as the target. The dataset represents historical Boston-area housing data; it is not Indian real-estate data.

### Tools
- Python, Pandas, NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook

### Workflow
1. Load and inspect data
2. Check missing values and duplicates
3. Prepare numeric features and target
4. Explore feature distributions and relationships
5. Split into training and testing sets
6. Train Linear Regression
7. Evaluate with MAE, MSE, RMSE and R²
8. Save predictions, metrics and visualizations

### Model Results
- Training rows: 404
- Testing rows: 102
- MAE: approximately 3.19
- RMSE: approximately 4.93
- R²: approximately 0.669

### Outputs
See the `outputs/` folder for cleaned data, model metrics, predictions and charts.

### Limitations
This model is educational and uses historical Boston-area data. It is not suitable for current house-price estimates, especially in India.

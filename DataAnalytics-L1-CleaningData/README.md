# Oasis Infobyte Data Analytics Internship
## Level 1 – Task 3: Data Cleaning

### Objective
Clean and validate the Titanic passenger dataset using Python and Pandas.

### Tools
- Python
- Pandas and NumPy
- Matplotlib
- Jupyter Notebook

### Workflow
1. Inspect the dataset and missing values.
2. Check and remove exact duplicate rows.
3. Standardise text columns.
4. Correct selected data types.
5. Handle missing Age, Fare, Embarked, and Cabin values.
6. Detect numeric outliers using the IQR method.
7. Compare data quality before and after cleaning.
8. Export the cleaned dataset and reports.

### Missing-value strategy
- Age: overall median.
- Fare: median within passenger class, then overall median.
- Embarked: mode.
- Cabin: `Unknown`, indicating unavailable source information.

### Outputs
- `titanic_cleaned.csv`
- `data_quality_report.csv`
- `before_after_summary.csv`
- `final_data_types.csv`
- `outlier_report.csv`
- Visualisation PNG files

### Dataset note
The source is the public Titanic dataset from Data Science Dojo's datasets repository. The source contains missing values but no exact duplicate rows. Outliers are reported rather than automatically removed because some extreme values may be valid observations.

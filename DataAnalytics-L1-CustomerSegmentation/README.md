# Customer Segmentation Analysis Using K-Means Clustering

## Internship
Oasis Infobyte (OIBSIP) - Data Analytics Internship

## Task
Level 1 - Customer Segmentation Analysis

## Objective
To segment retail customers based on purchasing behaviour using RFM analysis and K-Means clustering.

## Dataset
UCI Online Retail Dataset  
Source: https://archive.ics.uci.edu/dataset/352/online+retail

The dataset contains retail transactions from December 2010 to December 2011.

## Tools and Technologies
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Jupyter Notebook
- K-Means Clustering

## Project Workflow
1. Load and inspect the dataset.
2. Check missing values and duplicate records.
3. Clean customer and transaction data.
4. Calculate total transaction-line value.
5. Perform exploratory data analysis.
6. Calculate Recency, Frequency and Monetary (RFM) metrics.
7. Scale the features and use the Elbow Method.
8. Apply K-Means clustering with four clusters.
9. Analyse customer segments.
10. Prepare business recommendations.

## RFM Metrics
- Recency: Days since the customer's last purchase.
- Frequency: Number of unique invoices per customer.
- Monetary: Total purchase value per customer.

## Customer Segments
The clusters are interpreted using their RFM profiles:
- Occasional Customers
- High-Value Loyal Customers
- At-Risk / Inactive Customers
- Regular Customers

Cluster numbers are model-generated labels. Business-friendly names are assigned after reviewing the cluster profiles.

## Key Findings
- Cleaned transaction rows: approximately 392,692.
- Unique customers analysed: 4,338.
- Initial segmentation: four customer clusters.
- The results can help businesses plan targeted marketing and customer-retention campaigns.

## Limitations
The selection of four clusters is a practical modelling decision. Segment performance should be validated using real marketing campaign results before business implementation.

## How to Run
1. Download the Online Retail dataset from the UCI source above.
2. Place `Online Retail.xlsx` inside the `data/` directory.
3. Install dependencies using `pip install -r requirements.txt`.
4. Open `Customer_Segmentation_Analysis.ipynb` in Jupyter Notebook.
5. Run the notebook cells from top to bottom.

## Dataset Attribution
Dataset: Online Retail  
Source: UCI Machine Learning Repository  
https://archive.ics.uci.edu/dataset/352/online+retail

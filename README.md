# Customer-Churn-Prediction
Machine learning project to identify customers likely to churn using Logistic Regression. Covers the full data science workflow: data analysis, feature engineering, sklearn preprocessing pipelines, ROC-AUC evaluation, and joblib model export.

## About the Project
 
Customer churn is one of the most critical challenges in subscription-based and banking businesses. Losing a customer is far more expensive than retaining one. This project builds a machine learning system that identifies customers who are likely to churn, enabling businesses to take proactive retention action.
 
The pipeline covers every stage of a real-world ML project: data collection, exploratory analysis, feature engineering, preprocessing, model training, evaluation, and deployment-ready model export.
 
**Dataset Features:**
- `customer_id`, `credit_score`, `country`, `gender`, `age`, `tenure`
- `balance`, `products_number`, `credit_card`, `active_member`, `estimated_salary`
- Target: `churn` (0 = Retained, 1 = Churned)
---
 
## Project Pipeline
 
```
Data Collection → EDA → Feature Engineering → Preprocessing → Model Training → Evaluation → Deployment
```
 
| Step | Description |
|------|-------------|
| Step 1 | Library Setup & Imports |
| Step 2 | Data Collection (`customer_data.csv`) |
| Step 3 | Data Processing & Missing Value Analysis |
| Step 4 | Exploratory Data Analysis (EDA) |
| Step 5 | Feature Engineering |
| Step 6 | Preprocessing Pipeline (Scaling + Encoding) |
| Step 7 | Model Training (Logistic Regression) |
| Step 8 | Model Evaluation (Confusion Matrix, ROC-AUC) |
| Step 9 | Feature Importance Visualization |
| Step 10 | Model Export & Prediction |
 
---
 
## Exploratory Data Analysis
 
The EDA covers a thorough visual investigation of the dataset:
 
- **KDE Distribution plots** for numeric features (`credit_score`, `age`, `tenure`, `balance`, `products_number`, `estimated_salary`) split by churn status
- **Pairplot** of key numeric features colored by churn
- **Violin plot** of Age by Churn
- **Count plots** for all categorical features (`country`, `gender`, `credit_card`, `active_member`)
- **Donut charts** for gender-wise churn ratio
- **Correlation heatmap** of numeric features
- **Scatter plot** of Balance vs. Number of Products by Churn
- **Bar chart** of Churn Rate per number of products
---
 
## Feature Engineering
 
New features were derived to improve model signal:
 
| Feature | Description |
|---------|-------------|
| `balance_per_product` | Balance divided by number of products |
| `salary_balance_ratio` | Estimated salary to balance ratio |
| `age_group` | Age bucketed into ranges: `<25`, `25-34`, `35-44`, `45-54`, `55-64`, `65+` |
| `tenure_bucket` | Tenure grouped as: `0`, `1-2`, `3-5`, `6-10`, `10+` |
| `high_balance` | Binary flag for customers in the top 25% balance quartile |
 
---
 
## Model
 
**Algorithm:** Logistic Regression (`class_weight='balanced'`)
 
**Preprocessing pipeline:**
- Numeric features: Median imputation → Standard Scaling
- Categorical features: Most-frequent imputation → One-Hot Encoding
- Train/Test split: 70/30 with stratification on `churn`
**Evaluation metrics used:**
- Confusion Matrix
- Classification Report (Precision, Recall, F1)
- ROC-AUC Score
- ROC Curve Visualization
---
 
## Library Requirements
 
```
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
```
 
---
  
## Project Structure
 
```
Customer-Churn-Prediction/
│
├── Churn_Prediction.ipynb                    # Main notebook
├── customer_data.csv                # Input dataset
├── logistic_regression_model.pkl    # Saved trained model
├── feature_scaler.pkl               # Saved scaler
├── requirements.txt                 # Python dependencies
└── README.md
```
 
---

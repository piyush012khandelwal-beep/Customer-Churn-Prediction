
# Customer Churn Prediction

An end-to-end Machine Learning project that predicts
customer churn and provides actionable retention recommendations.

## Dataset

Source: Kaggle - Telco Customer Churn IBM Dataset

https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset

The dataset is used for educational and machine learning purposes.
The dataset file is not included in this repository.

## Features

- Customer churn prediction
- Logistic Regression and Random Forest comparison
- Customer risk segmentation
- High-risk customer identification
- Automated retention recommendations
- Executive business dashboard
- Revenue risk analysis
- Batch customer prediction
- Downloadable CSV reports

## Tech Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Matplotlib and Seaborn
- Joblib

## Project Structure

- app/: Streamlit application
- models/: Trained ML model
- data/: Dataset
- reports/: Generated reports
- notebooks/: Model development

## Model

Random Forest Classifier with a preprocessing pipeline
and a tuned probability threshold.

## How to Run Locally

```bash
pip install -r requirements.txt
python -m streamlit run app/app.py
```
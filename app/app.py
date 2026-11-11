import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction")
st.caption("Machine Learning Powered Customer Retention System")

# -----------------------------------
# LOAD SAVED MODEL
# -----------------------------------

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "churn_rf_bundle.joblib"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

if not MODEL_PATH.exists():
    st.error("Model file not found. Save the trained Random Forest model first.")
    st.stop()

bundle = load_model()

model = bundle["model"]
threshold = bundle["threshold"]
high_charge_cutoff = bundle["high_charge_cutoff"]

# -----------------------------------
# DATA LOADING
# -----------------------------------
# Load original dataset


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR, "data", "Telco_customer_churn (1).xlsx"
)

@st.cache_data
def load_data():
    return pd.read_excel(DATA_PATH)


if os.path.exists(DATA_PATH):
    df = load_data()
else:
    st.warning("Please upload the Telco Customer Churn dataset.")

    uploaded_dataset = st.file_uploader(
        "Upload Telco Customer Churn Excel File",
        type=["xlsx", "xls"]
    )

    if uploaded_dataset is None:
        st.info("Upload the Excel dataset to access the analytics dashboard.")
        st.stop()

    df = pd.read_excel(uploaded_dataset)

df.columns = df.columns.str.strip()

# =====================================
# DATASET OVERVIEW
# =====================================

st.subheader("Dataset Overview")

st.write("Dataset Shape:", df.shape)

st.write("Column Names:")
st.write(df.columns.tolist())

st.write("Dataset Preview:")
st.dataframe(df.head(10), width=True)

# =====================================
# PROFESSIONAL KPI CARDS
# =====================================

st.title("Customer Churn Analytics Dashboard")

total_customers = len(df)

churned_customers = (
    df["Churn Label"].astype(str).str.strip() == "Yes"
).sum()

retained_customers = total_customers - churned_customers

churn_rate = (churned_customers / total_customers) * 100

avg_monthly_charges = pd.to_numeric(
    df["Monthly Charges"], errors="coerce"
).mean()

st.subheader("Business Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

with col2:
    st.metric(
        "Churned Customers",
        f"{churned_customers:,}"
    )

with col3:
    st.metric(
        "Retained Customers",
        f"{retained_customers:,}"
    )

with col4:
    st.metric(
        "Churn Rate",
        f"{churn_rate:.2f}%"
    )

st.metric(
    "Average Monthly Charges",
    f"${avg_monthly_charges:.2f}"
)


# =====================================
# CUSTOMER CHURN DISTRIBUTION
# =====================================

# Clean column names
df.columns = df.columns.str.strip()

# Correct target column
target_col = "Churn Label"

if target_col not in df.columns:
    st.error("Churn Label column not found!")
    st.write(df.columns.tolist())
    st.stop()

# Churn analysis
st.subheader("Customer Churn Distribution")

churn_counts = df[target_col].value_counts()

col1, col2 = st.columns(2)

with col1:
    st.metric("Total Customers", len(df))

with col2:
    churn_rate = (
        df[target_col].astype(str).str.strip().eq("Yes").mean() * 100
    )
    st.metric("Churn Rate", f"{churn_rate:.2f}%")

st.write("Churn Counts:")
st.write(churn_counts)


st.subheader("Churn Analysis by Contract")

contract_churn = pd.crosstab(
    df["Contract"],
    df["Churn Label"],
    normalize="index"
) * 100

st.bar_chart(contract_churn)


st.subheader("Churn Analysis by Payment Method")

payment_churn = pd.crosstab(
    df["Payment Method"],
    df["Churn Label"],
    normalize="index"
) * 100

st.bar_chart(payment_churn)

st.subheader("Churn Analysis by Tenure")

df["Tenure Months"] = pd.to_numeric(
    df["Tenure Months"],
    errors="coerce"
)

df["Tenure Group"] = pd.cut(
    df["Tenure Months"],
    bins=[-1, 12, 24, 48, 60, 72],
    labels=[
        "0-12 Months",
        "13-24 Months",
        "25-48 Months",
        "49-60 Months",
        "61-72 Months"
    ]
)

tenure_churn = pd.crosstab(
    df["Tenure Group"],
    df["Churn Label"],
    normalize="index"
) * 100

st.bar_chart(tenure_churn)
# -----------------------------------
# FEATURE DEFINITIONS
# -----------------------------------

categorical_features = [
    'Gender',
    'Senior Citizen',
    'Partner',
    'Dependents',
    'Phone Service',
    'Multiple Lines',
    'Internet Service',
    'Online Security',
    'Online Backup',
    'Device Protection',
    'Tech Support',
    'Streaming TV',
    'Streaming Movies',
    'Contract',
    'Paperless Billing',
    'Payment Method'
]

numerical_features = [
    'Tenure Months',
    'Monthly Charges',
    'Total Charges'
]

feature_columns = (
    categorical_features + numerical_features
)


# =====================================
# CUSTOMER RISK SEGMENTATION
# =====================================

st.subheader("Customer Risk Segmentation")

# Check required columns
missing = [
    col for col in feature_columns
    if col not in df.columns
]

if missing:
    st.error(f"Missing model features: {missing}")
    st.stop()

# Predict churn probabilities
# Prepare data for churn risk prediction
risk_data = df[feature_columns].copy()

# Remove leading/trailing spaces from text columns
for col in risk_data.select_dtypes(include=["object"]).columns:
    risk_data[col] = risk_data[col].astype(str).str.strip()

# Replace empty strings with NaN
risk_data = risk_data.replace(r"^\s*$", np.nan, regex=True)

# Convert numeric columns safely
numeric_columns = [
    "Senior Citizen",
    "Tenure Months",
    "Monthly Charges",
    "Total Charges"
]

for col in numeric_columns:
    risk_data[col] = pd.to_numeric(risk_data[col], errors="coerce")

risk_data["Churn Probability"] = model.predict_proba(
    risk_data[feature_columns]
)[:, 1]

# Classify risk
risk_data["Risk Category"] = "Low Risk"

risk_data.loc[
    risk_data["Churn Probability"] >= threshold,
    "Risk Category"
] = "Medium Risk"

risk_data.loc[
    risk_data["Churn Probability"] >= 0.50,
    "Risk Category"
] = "High Risk"

st.subheader("Executive Business Insights")

total_monthly_revenue = pd.to_numeric(
    df["Monthly Charges"], errors="coerce"
).sum()

churned_revenue = pd.to_numeric(
    df.loc[
        df["Churn Label"].astype(str).str.strip() == "Yes",
        "Monthly Charges"
    ],
    errors="coerce"
).sum()

revenue_at_risk = pd.to_numeric(
    risk_data.loc[
        risk_data["Risk Category"] == "High Risk",
        "Monthly Charges"
    ],
    errors="coerce"
).sum()

col1, col2, col3 = st.columns(3)

col1.metric("Total Monthly Revenue", f"${total_monthly_revenue:,.2f}")
col2.metric("Revenue Lost from Churned Customers", f"${churned_revenue:,.2f}")
col3.metric("Monthly Revenue at High Risk", f"${revenue_at_risk:,.2f}")

st.subheader("Revenue Overview")

revenue_comparison = pd.DataFrame({
    "Category": ["Active Customers", "Churned Customers", "High-Risk Customers"],
    "Monthly Revenue": [
        total_monthly_revenue - churned_revenue,
        churned_revenue,
        revenue_at_risk
    ]
})

st.bar_chart(revenue_comparison.set_index("Category"), y="Monthly Revenue")

# Count customers
risk_counts = (
    risk_data["Risk Category"]
    .value_counts()
    .reindex(
        ["Low Risk", "Medium Risk", "High Risk"],
        fill_value=0
    )
)

# Display metrics
col1, col2, col3 = st.columns(3)

col1.metric("Low Risk", int(risk_counts["Low Risk"]))
col2.metric("Medium Risk", int(risk_counts["Medium Risk"]))
col3.metric("High Risk", int(risk_counts["High Risk"]))

# Display chart
st.bar_chart(risk_counts)

# Customer risk table
st.subheader("High-Risk Customer Analysis")




st.subheader("Download Customer Risk Report")

# Create a separate export copy
export_data = risk_data.copy()

# Convert churn probability to percentage
export_data["Churn Probability"] = (
    export_data["Churn Probability"] * 100
).round(2)

# Sort by highest churn probability
export_data = export_data.sort_values(
    by="Churn Probability",
    ascending=False
)

# Convert report to CSV
csv_report = export_data.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Complete Customer Risk Report",
    data=csv_report,
    file_name="customer_churn_risk_report.csv",
    mime="text/csv",
    width="stretch"
)


def generate_recommendations(row):

    recommendations = []

    # 1. Risk-based recommendations
    if row["Risk Category"] == "High Risk":
        recommendations.append("Priority retention outreach")

    elif row["Risk Category"] == "Medium Risk":
        recommendations.append("Proactive customer check-in")

    else:
        recommendations.append("Regular loyalty engagement")

    # 2. Contract recommendations
    if row["Contract"] == "Month-to-month":
        recommendations.append("Offer annual contract incentive")

    # 3. New customer support
    tenure = pd.to_numeric(
        pd.Series([row["Tenure Months"]]),
        errors="coerce"
    ).iloc[0]

    if pd.notna(tenure) and tenure <= 12:
        recommendations.append("Provide onboarding support")

    # 4. Technical support
    if (
        row["Internet Service"] != "No"
        and row["Tech Support"] == "No"
    ):
        recommendations.append("Offer tech support trial")

    # 5. High monthly charges
    monthly_charges = pd.to_numeric(
        pd.Series([row["Monthly Charges"]]),
        errors="coerce"
    ).iloc[0]

    charge_cutoff = pd.to_numeric(
        pd.Series([df["Monthly Charges"].quantile(0.75)]),
        errors="coerce"
    ).iloc[0]

    if (
        pd.notna(monthly_charges)
        and monthly_charges >= charge_cutoff
    ):
        recommendations.append("Review plan and offer savings")

    # 6. Payment method
    if row["Payment Method"] == "Electronic check":
        recommendations.append("Encourage automatic payment")

    # 7. Online security
    if (
        row["Internet Service"] != "No"
        and row["Online Security"] == "No"
    ):
        recommendations.append("Offer optional security package")

    return " | ".join(recommendations)

risk_data["Retention Recommendations"] = risk_data.apply(
    generate_recommendations,
    axis=1
)
# Add customer ID to risk data
if "Customer ID" in df.columns:
    risk_data.insert(0, "Customer ID", df["Customer ID"].values)

# Sort customers by churn probability
risk_data = risk_data.sort_values(
    by="Churn Probability",
    ascending=False
)

# Select useful business columns
display_columns = [
    "Customer ID",
    "Churn Probability",
    "Risk Category",
    "Contract",
    "Tenure Months",
    "Monthly Charges",
    "Retention Recommendations"
]

# Keep only available columns
display_columns = [
    col for col in display_columns
    if col in risk_data.columns
]

# Display top 20 customers
top_customers = risk_data[display_columns].head(20).copy()

top_customers["Churn Probability"] = (
    top_customers["Churn Probability"] * 100
).round(2)

st.dataframe(
    top_customers,
    width="stretch",
    hide_index=True
)
# -----------------------------------
# RISK SEGMENTATION
# -----------------------------------

def get_risk_segment(score):

    if score < threshold:
        return "Low Risk"

    elif score < 0.50:
        return "Medium Risk"

    return "High Risk"


# -----------------------------------
# RETENTION RECOMMENDATION ENGINE
# -----------------------------------

def recommend_retention(row, risk):

    actions = []

    if risk == "High Risk":
        actions.append(
            "Priority retention outreach"
        )

    elif risk == "Medium Risk":
        actions.append(
            "Proactive customer check-in"
        )

    else:
        actions.append(
            "Regular loyalty engagement"
        )

    if row["Contract"] == "Month-to-month":
        actions.append(
            "Offer annual contract incentives"
        )

    if row["Tenure Months"] <= 12:
        actions.append(
            "Provide onboarding support"
        )

    if (
        row["Internet Service"] != "No"
        and row["Tech Support"] == "No"
    ):
        actions.append(
            "Offer technical support trial"
        )

    if row["Monthly Charges"] >= high_charge_cutoff:
        actions.append(
            "Review plan and offer suitable savings"
        )

    if row["Payment Method"] == "Electronic check":
        actions.append(
            "Offer convenient automatic payment options"
        )

    if (
        row["Internet Service"] != "No"
        and row["Online Security"] == "No"
    ):
        actions.append(
            "Offer optional online security package"
        )

    return actions[:3]


# -----------------------------------
# DASHBOARD TABS
# -----------------------------------

tab1, tab2, tab3 = st.tabs([
    "🏠 Dashboard",
    "🔍 Customer Prediction",
    "📁 Batch Prediction"
])


# ===================================
# TAB 1: DASHBOARD
# ===================================

with tab1:

    st.subheader("Model Performance Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", "64.87%")
    col2.metric("Churn Recall", "93.05%")
    col3.metric("F1 Score", "58.44%")
    col4.metric("ROC AUC", "0.8511")

    st.divider()

    st.subheader("Customer Risk Segmentation")

    risk_data = pd.DataFrame({
        "Risk Segment": [
            "Low Risk",
            "Medium Risk",
            "High Risk"
        ],
        "Customers": [290, 592, 527]
    })

    col1, col2 = st.columns(2)

    with col1:
        st.bar_chart(
            risk_data.set_index("Risk Segment")
        )

    with col2:
        st.dataframe(
            risk_data,
            width=True,
            hide_index=True
        )

    st.info(
        "Model evaluation metrics are from the held-out "
        "test dataset. Risk segment counts are from the "
        "test-set segmentation completed in this project."
    )

    st.subheader("Business Objective")

    st.write(
        """
        This application predicts potential telecom customer
        churn and generates personalized retention suggestions.

        The main objective is to identify at-risk customers
        early and help businesses prioritize retention efforts.
        """
    )


# ===================================
# TAB 2: SINGLE CUSTOMER PREDICTION
# ===================================

with tab2:

    st.subheader("Enter Customer Information")

    with st.form("customer_form"):

        col1, col2, col3 = st.columns(3)

        with col1:

            gender = st.selectbox(
                "Gender",
                ["Female", "Male"]
            )

            senior = st.selectbox(
                "Senior Citizen",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            partner = st.selectbox(
                "Partner",
                ["Yes", "No"]
            )

            dependents = st.selectbox(
                "Dependents",
                ["Yes", "No"]
            )

            phone = st.selectbox(
                "Phone Service",
                ["Yes", "No"]
            )

            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["No", "Yes", "No phone service"]
            )

        with col2:

            internet = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"]
            )

            online_security = st.selectbox(
                "Online Security",
                ["Yes", "No", "No internet service"]
            )

            online_backup = st.selectbox(
                "Online Backup",
                ["Yes", "No", "No internet service"]
            )

            device_protection = st.selectbox(
                "Device Protection",
                ["Yes", "No", "No internet service"]
            )

            tech_support = st.selectbox(
                "Tech Support",
                ["Yes", "No", "No internet service"]
            )

            streaming_tv = st.selectbox(
                "Streaming TV",
                ["Yes", "No", "No internet service"]
            )

            streaming_movies = st.selectbox(
                "Streaming Movies",
                ["Yes", "No", "No internet service"]
            )

        with col3:

            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"]
            )

            paperless = st.selectbox(
                "Paperless Billing",
                ["Yes", "No"]
            )

            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )

            tenure = st.number_input(
                "Tenure Months",
                min_value=0,
                max_value=72,
                value=12
            )

            monthly = st.number_input(
                "Monthly Charges",
                min_value=0.0,
                max_value=200.0,
                value=70.0
            )

            total = st.number_input(
                "Total Charges",
                min_value=0.0,
                value=840.0
            )

        submitted = st.form_submit_button(
            "Predict Customer Churn",
            use_container_width=True
        )

    if submitted:

        customer = pd.DataFrame([{
            "Gender": gender,
            "Senior Citizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "Phone Service": phone,
            "Multiple Lines": multiple_lines,
            "Internet Service": internet,
            "Online Security": online_security,
            "Online Backup": online_backup,
            "Device Protection": device_protection,
            "Tech Support": tech_support,
            "Streaming TV": streaming_tv,
            "Streaming Movies": streaming_movies,
            "Contract": contract,
            "Paperless Billing": paperless,
            "Payment Method": payment,
            "Tenure Months": tenure,
            "Monthly Charges": monthly,
            "Total Charges": total
        }])

        score = model.predict_proba(
            customer[feature_columns]
        )[0, 1]

        risk = get_risk_segment(score)

        prediction = (
            "Churn" if score >= threshold
            else "No Churn"
        )

        recommendations = recommend_retention(
            customer.iloc[0],
            risk
        )

        st.divider()

        st.subheader("Prediction Results")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Estimated Churn Score",
            f"{score * 100:.2f}%"
        )

        col2.metric(
            "Predicted Class",
            prediction
        )

        col3.metric(
            "Risk Segment",
            risk
        )

        st.subheader("Retention Recommendations")

        for action in recommendations:
            st.write(f"✅ {action}")

        st.caption(
            "The churn score is a model estimate, not "
            "a guarantee of actual customer behaviour."
        )


# ===================================
# TAB 3: BATCH CUSTOMER PREDICTION
# ===================================

with tab3:

    st.subheader("Upload Customer CSV")

    st.write(
        "Upload a CSV containing the same 19 input "
        "features used during model training."
    )

    uploaded_file = st.file_uploader(
        "Choose Customer CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        batch_df = pd.read_csv(uploaded_file)

        missing_columns = [
            col for col in feature_columns
            if col not in batch_df.columns
        ]

        if missing_columns:

            st.error(
                f"Missing columns: {missing_columns}"
            )

        elif batch_df.empty:

            st.error("The uploaded CSV has no records.")

        else:

            for col in numerical_features:
                batch_df[col] = pd.to_numeric(
                    batch_df[col],
                    errors="coerce"
                )

            batch_df["Senior Citizen"] = pd.to_numeric(
                batch_df["Senior Citizen"],
                errors="coerce"
            )

            for col in categorical_features:
                if col != "Senior Citizen":
                    batch_df[col] = (
                        batch_df[col]
                        .astype("string")
                        .str.strip()
                    )

            if batch_df["Senior Citizen"].isna().any():
                st.error(
                    "Senior Citizen must contain 0 or 1."
                )
                st.stop()

            probabilities = model.predict_proba(
                batch_df[feature_columns]
            )[:, 1]

            batch_results = batch_df.copy()

            batch_results["Churn Score"] = probabilities

            batch_results["Predicted Churn"] = [
                "Yes" if score >= threshold else "No"
                for score in probabilities
            ]

            batch_results["Risk Segment"] = [
                get_risk_segment(score)
                for score in probabilities
            ]

            batch_results["Retention Recommendations"] = [
                " | ".join(
                    recommend_retention(
                        row,
                        get_risk_segment(score)
                    )
                )
                for (_, row), score in zip(
                    batch_df.iterrows(),
                    probabilities
                )
            ]

            st.success("Predictions generated successfully!")

            st.subheader("Customer Prediction Results")

            st.dataframe(
                batch_results,
                use_container_width=True
            )

            st.download_button(
                "Download Prediction Report",
                data=batch_results.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="churn_predictions.csv",
                mime="text/csv"
            )






            
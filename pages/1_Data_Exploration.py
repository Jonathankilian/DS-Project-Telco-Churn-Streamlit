import streamlit as st
import pandas as pd
# import numpy as np
from pathlib import Path
from config import CSV_FILE, TARGET_COLUMN, load_data, get_schema_df
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


st.set_page_config(page_title="Data Exploration", page_icon="🔎", layout="wide")

df = load_data()
schema_df = get_schema_df()



st.title("🔎 Data Exploration")
st.subheader("Explore the Telco Churn Dataset")


@st.cache_data
def load_data():
    """Load and cache data"""
    return pd.read_csv(CSV_FILE)


df = load_data()

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Dataset", "Data Quality", "📊 Statistics"])

with tab1:
    st.header("🎯 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Rows", df.shape[0])
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Churn Rate", f"{(df[TARGET_COLUMN].sum() / len(df) * 100):.2f}%")
    col4.metric("Non-Churn Rate", f"{((1 - df[TARGET_COLUMN].mean()) * 100):.2f}%")

    churn_counts = df[TARGET_COLUMN].value_counts()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn Distribution")
        st.bar_chart(churn_counts)

    with col2:
        st.subheader("Churn Statistics")
        st.write(f"**Churned Customers**: {churn_counts[True]}")
        st.write(f"**Active Customers**: {churn_counts[False]}")

with tab2:
    st.header("📋 Dataset Preview")
    with st.expander("🔎 **Dataset Column Meaning:**"):
        col_meaning_data = [
            ["male", "Customer gender indicator (1 = male, 0 = female).", "männlich"],
            ["SeniorCitizen", "Shows if the customer is a senior citizen (1 = yes, 0 = no).", "Senior"],
            ["Partner", "Shows if the customer has a partner (Yes or No).", "Partner"],
            ["Dependents", "Shows if the customer has people who depend on them, for example children (Yes or No).",
             "Abhängige"],
            ["tenure", "Number of months the customer has been with the company.", "Vertragsdauer"],
            ["PhoneService", "Shows if the customer has phone service (Yes or No).", "Telefondienst"],
            ["MultipleLines", "Shows if the customer uses more than one phone line (Yes, No, or No phone service).",
             "Mehrere Leitungen"],
            ["InternetService", "Type of internet service used by the customer (DSL, Fiber optic, or No).",
             "Internetdienst"],
            ["OnlineSecurity", "Shows if online security service is included (Yes, No, or No internet service).",
             "Online-Sicherheit"],
            ["OnlineBackup", "Shows if online backup service is included (Yes, No, or No internet service).",
             "Online-Backup"],
            ["DeviceProtection", "Shows if device protection service is included (Yes, No, or No internet service).",
             "Geräteschutz"],
            ["TechSupport", "Shows if technical support service is included (Yes, No, or No internet service).",
             "Technischer Support"],
            ["StreamingTV", "Shows if TV streaming service is included (Yes, No, or No internet service).",
             "Streaming TV"],
            ["StreamingMovies", "Shows if movie streaming service is included (Yes, No, or No internet service).",
             "Streaming Filme"],
            ["Contract", "Type of customer contract (Month-to-month, One year, or Two year).", "Vertragsart"],
            ["PaperlessBilling", "Shows if the customer uses paperless billing (Yes or No).", "Papierlose Abrechnung"],
            ["PaymentMethod", "Payment method used by the customer.", "Zahlungsmethode"],
            ["MonthlyCharges", "Amount the customer pays each month.", "Monatliche Kosten"],
            ["TotalCharges", "Total amount the customer has paid so far.", "Gesamtkosten"],
            ["Churn", "Shows if the customer has left the company (Yes or No).", "Abwanderung"],
            ["tenure_category", "Customer group based on how long they have been with the company.",
             "Vertragsdauer Kategorie"],
            ["AvgMonthly", "Average amount paid per month, calculated from total charges and tenure.",
             "Durchschnittliche Monatskosten"],
            ["IsNewCustomer", "Shows if the customer is considered new, based on tenure.", "Neukunde"],
            ["HasAddOns", "Shows if the customer uses at least one additional service.", "Zusatzdienste"],
        ]

        col_meaning_df = pd.DataFrame(
            col_meaning_data,
            columns=[
                "Column Name",
                "Description",
                "Column Name (German)"
            ]
        )

        st.dataframe(col_meaning_df, use_container_width=True)

    st.dataframe(df.head(10), use_container_width=True)

    st.header("Meta-Data-View-Table")
    st.dataframe(schema_df)

    with st.expander("🔎 **Meta View Column Meaning:**"):
        col_meaning_meta_data = [
            ["dtype", "Shows the data type of the column, for example number, text, or boolean."],
            ["semantic_type",
             "Shows the meaning of the column based on its values, for example category or numeric value."],
            ["n_unique", "Shows how many different values exist in the column."],
            ["cardinality_ratio",
             "≈ 0.0 → many repeated values (e.g. categories); ≈ 1.0 → almost every value is unique (e.g. IDs)"],
            ["unique_values", "Shows a list or sample of different values found in the column."],
            ["n_isna", "Shows how many values are missing (null or NaN)."],
            ["empty_string_count", "Shows how many values are empty strings with no text."],
            ["whitespace_only_count", "Shows how many values contain only spaces or tabs."],
            ["missing_pct", "Shows the percentage of missing values in the column."],
            ["all_null", "Shows if all values in the column are missing. True means no real values exist."],
            ["constant", "Shows if all values in the column are the same."]
        ]
        col_meaning_meta_df = pd.DataFrame(col_meaning_meta_data,
            columns=[
                "Column Name",
                "Description",
            ])
        st.dataframe(col_meaning_meta_df, use_container_width=True)


with tab3:

    st.header("🔍 Data Quality")
    st.subheader("Missing Values")
    total_missing_pct = schema_df["missing_pct"].sum()
    if total_missing_pct == 0:
        st.success("✅ No missing values found!")
    else:
        st.error("❌ missing values found")

    col1, col2, col3 = st.columns(3)
    col1.metric("Count NaN or NULL:", schema_df["n_isna"].sum())
    col2.metric("Count empty String", schema_df["empty_string_count"].sum())
    col3.metric("Count Whitespaces only", schema_df["whitespace_only_count"].sum())

with tab4:
    st.header("📊 Statistics")
    st.subheader("statistical Overview")
    st.write(" → bools temporär transformiert [0,1]")
    df_num = df.copy()

    bool_cols = df_num.select_dtypes(include="bool").columns
    df_num[bool_cols] = df_num[bool_cols].astype(int)

    st.dataframe(df_num.describe())

    st.subheader("Categorical Variables")
    CATEGORICAL_TYPES = {"category", "category_str"}
    cat_cols = schema_df.loc[schema_df["semantic_type"].isin(CATEGORICAL_TYPES), "columnname"].tolist()
    if not cat_cols:
        st.info("Keine kategorischen Spalten gefunden (semantic_type = category / category_str).")

    else:
        cat_cols_sorted = (
            schema_df.loc[schema_df["columnname"].isin(cat_cols), ["columnname", "n_unique"]]
            .sort_values("n_unique")
            ["columnname"]
            .tolist()
        )

        for col in cat_cols_sorted:
            with st.expander(f"📊 {col}", expanded=False):
                s = df[col].astype("object")
                s = s[~s.isna()]
                if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
                    s = s[~(s.astype(str).str.strip() == "")]

                vc = s.value_counts(dropna=False)
                out = vc.rename_axis(col).reset_index(name="count")

                # Limit output to top-K categories for readability
                TOP_K = 50
                if len(out) > TOP_K:
                    st.caption(f"Zeige Top {TOP_K} von {len(out)} Kategorien")
                    out = out.head(TOP_K)

                st.dataframe(out, use_container_width=True, hide_index=True)
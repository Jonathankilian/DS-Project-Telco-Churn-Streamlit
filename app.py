import streamlit as st
#import pandas as pd
from config import load_data


# Page Config
st.set_page_config(page_title='Telco Churn Analysis & Prediction', page_icon='📱', layout='wide', initial_sidebar_state='expanded')


def main():
    # Titel
    st.title('Telco Churn Analysis & Prediction')
    st.write('Interactive analysis and machine learning predictions for costumer churn behaviour')
    st.write("___")
    # Daten laden
    df = load_data()
    col1, col2 = st.columns(2)
    with col1:
        st.header("📚 Contents")
        st.markdown("""
        🔎 **Data Exploration** – Inspecting and cleaning the dataset

        📊 **Visualization** – Exploratory Data Analysis (EDA)

        🤖 **ML Prediction** – Machine learning–based predictions

        👈 Use the sidebar for navigation.
        """)
    with col2:
        st.header("ℹ️ About")

        st.markdown("""
        **Project Context**  
        Academic data science project using Python (Berliner Hochschule für Technik).

        **Author**  
        Jonathan Wirtz

        **Publication**  
        January 2026

        **Dataset**  
        [Kaggle – Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
        """)

    st.write("___")
    st.header("🎯 Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", df.shape[0])
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Churn Rate (Kündigungsrate)",f"{(df['Churn'].sum()/len(df) * 100):.2f}%")

    #st.selectbox("Überblick Daten")
    with st.expander("🛢 Data Preview"):
        col1, col2 = st.columns(2)
        with col1:
            num_col = st.slider("Anzahl Spalten", 5, df.shape[1])
        with col2:
            num_row = st.slider("Anzahl Zeilen", 5, 40)
        df_view = df.iloc[:num_row, : num_col]
        st.dataframe(data=df_view)


if __name__ == '__main__':
    main()
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib
import os

# Set page configuration
st.set_page_config(page_title="AutoML App", layout="wide")

# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = "Data Ingestion"

# Sidebar with navigation
st.sidebar.title("Navigation")
with st.sidebar:
    if st.button("Data Ingestion"):
        st.session_state.page = "Data Ingestion"
    if st.button("Data Transformation"):
        st.session_state.page = "Data Transformation"
    if st.button("Auto Train ML models"):
        st.session_state.page = "Auto Train ML models"
    if st.button("Freeze the learnings"):
        st.session_state.page = "Freeze the learnings"

# Page navigation logic
if st.session_state.page == "Data Ingestion":
    st.header("Data Ingestion")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        if st.button('Ingest'):
            st.session_state.data = pd.read_csv(uploaded_file)
            st.write(st.session_state.data.head())
            st.write(f"Number of rows: {st.session_state.data.shape[0]}")
            st.write(f"Number of columns: {st.session_state.data.shape[1]}")
            st.success("Data successfully ingested!")

elif st.session_state.page == "Data Transformation":
    st.header("Data Transformation")
    if st.session_state.data is not None:
        # Remove features
        features_to_remove = st.multiselect("Select features to remove", st.session_state.data.columns)
        if features_to_remove:
            if st.button("Delete selected feature?"):
                st.session_state.data = st.session_state.data.drop(columns=features_to_remove)
                st.success("Selected features removed.")
                st.write(f"Number of rows: {st.session_state.data.shape[0]}")
                st.write(f"Number of columns: {st.session_state.data.shape[1]}")
        
        # Remove rows with null values based on selected column
        column_to_check_null = st.selectbox("Select column to check for null values", st.session_state.data.columns)
        if st.button("Remove rows with null values in selected column"):
            st.session_state.data = st.session_state.data.dropna(subset=[column_to_check_null])
            st.success(f"Rows with null values in '{column_to_check_null}' removed.")
            st.write(f"Number of rows: {st.session_state.data.shape[0]}")
            st.write(f"Number of columns: {st.session_state.data.shape[1]}")

        # Convert selected column to numbers
        column_to_convert = st.selectbox("Select column to convert to numbers", st.session_state.data.columns)
        if st.button("Convert selected column to numbers"):
            st.session_state.data[column_to_convert] = pd.to_numeric(st.session_state.data[column_to_convert], errors='coerce')
            st.success(f"'{column_to_convert}' converted to numeric.")
        
        st.write(st.session_state.data.head())
    else:
        st.warning("Please ingest data first.")

elif st.session_state.page == "Auto Train ML models":
    st.header("Auto Train ML models")
    if st.session_state.data is not None:
        target = st.selectbox("Select target variable", st.session_state.data.columns)
        test_size = st.slider("Select test size", 0.1, 0.5, 0.2)
        
        if st.button("Train Models"):
            X = st.session_state.data.drop(columns=[target])
            y = st.session_state.data[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
            
            # Decision Tree
            dt = DecisionTreeClassifier()
            dt.fit(X_train, y_train)
            dt_score = accuracy_score(y_test, dt.predict(X_test))
            st.write(f"Decision Tree Accuracy: {dt_score:.2f}")
            
            # AdaBoost
            ada = AdaBoostClassifier()
            ada.fit(X_train, y_train)
            ada_score = accuracy_score(y_test, ada.predict(X_test))
            st.write(f"AdaBoost Accuracy: {ada_score:.2f}")
            
            # Linear Regression (if target is continuous)
            if st.session_state.data[target].dtype in ['float64', 'int64']:
                lr = LinearRegression()
                lr.fit(X_train, y_train)
                lr_score = mean_squared_error(y_test, lr.predict(X_test))
                st.write(f"Linear Regression MSE: {lr_score:.2f}")
            
            st.success("Models trained successfully!")
    else:
        st.warning("Please ingest data first.")

elif st.session_state.page == "Freeze the learnings":
    st.header("Freeze the learnings")
    if st.session_state.data is not None:
        export_path = st.text_input("Enter export path", "models")
        if st.button("Export Models"):
            # Here you would typically save your trained models
            # For this example, we'll just create dummy models
            dummy_dt = DecisionTreeClassifier()
            dummy_ada = AdaBoostClassifier()
            dummy_lr = LinearRegression()
            
            os.makedirs(export_path, exist_ok=True)
            joblib.dump(dummy_dt, os.path.join(export_path, "decision_tree.joblib"))
            joblib.dump(dummy_ada, os.path.join(export_path, "adaboost.joblib"))
            joblib.dump(dummy_lr, os.path.join(export_path, "linear_regression.joblib"))
            
            st.success(f"Models exported to {export_path}")
    else:
        st.warning("Please ingest and train models first.")

# Footer
st.sidebar.markdown("---")
st.sidebar.text("AutoML App v1.0")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import scipy.stats as stats

def load_dataset(file):
    """Load dataset from a CSV file and clean data."""
    df = pd.read_csv(file)
    st.write("### Raw Data Preview:")
    st.write(df.head())
    
    df.replace("?", np.nan, inplace=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].nunique() <= 10:
            df = pd.get_dummies(df, columns=[col], drop_first=True)
        else:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
    
    st.write("### Cleaned Data Preview:")
    st.write(df.head())
    
    if df.empty or df.isnull().all().all():
        st.error("Dataset is empty or contains only missing values after cleaning. Please upload a valid dataset.")
        return None
    
    return df

def perform_random_forest(df, target_col, n_estimators=100, max_depth=None):
    """Perform Random Forest Regression on the dataset."""
    if df is None or df.empty:
        st.error("Cannot perform regression on an empty dataset.")
        return None, None, None, None, None, None, None, None
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    if X.empty or y.empty:
        st.error("No valid features or target column found. Please check the dataset.")
        return None, None, None, None, None, None, None, None
    
    if len(df) < 2:
        st.error("Not enough data points for regression. Please upload a larger dataset.")
        return None, None, None, None, None, None, None, None
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, mse, r2, X_train, y_train, X_test, y_test, y_pred

def plot_feature_importance(model, X):
    """Plot feature importance for Random Forest."""
    if model is None or X is None:
        return
    
    importance = model.feature_importances_
    features = X.columns
    
    fig, ax = plt.subplots()
    sns.barplot(x=importance, y=features, ax=ax)
    ax.set_title("Feature Importance")
    st.pyplot(fig)

def analyze_regression_results(y_test, y_pred):
    """Perform additional analysis on regression results."""
    residuals = y_test - y_pred
    
    st.write("### Residual Analysis")
    fig, ax = plt.subplots()
    sns.histplot(residuals, kde=True, ax=ax)
    ax.set_title("Residual Distribution")
    st.pyplot(fig)
    
    fig, ax = plt.subplots()
    stats.probplot(residuals, dist="norm", plot=ax)
    st.pyplot(fig)
    
    fig, ax = plt.subplots()
    ax.scatter(y_pred, residuals)
    ax.axhline(y=0, color='red', linestyle='--')
    ax.set_xlabel("Predicted Values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs. Predicted Values")
    st.pyplot(fig)

st.title("Random Forest Regression App")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file:
    df = load_dataset(uploaded_file)
    
    if df is not None:
        target_col = st.selectbox("Select the target column", df.columns)
        n_estimators = st.slider("Select number of trees", 10, 200, value=100)
        max_depth = st.slider("Select max depth of trees", 1, 20, value=5)
        
        if st.button("Run Random Forest Regression"):
            model, mse, r2, X_train, y_train, X_test, y_test, y_pred = perform_random_forest(df, target_col, n_estimators, max_depth)
            
            if model is not None:
                st.write(f"Mean Squared Error: {mse}")
                st.write(f"R-squared Score: {r2}")
                plot_feature_importance(model, X_train)
                analyze_regression_results(y_test, y_pred)


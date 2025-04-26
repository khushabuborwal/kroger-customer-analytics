import pandas as pd
import numpy as np
import joblib
import pyodbc
import config
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def load_data():
    """
    Load Transactions and Households tables from Azure SQL via pyodbc.
    """
    conn = pyodbc.connect(config.ODBC_CONN_STR)
    
    # Get transactions data
    tx = pd.read_sql("SELECT * FROM Transactions;", conn)
    
    # Get household data
    households = pd.read_sql("SELECT * FROM Households;", conn)
    
    conn.close()
    return tx, households

def prepare_features(tx, households):
    """
    Prepare features for churn prediction:
    - Frequency: Number of transactions in the period
    - Monetary: Total spend amount
    - Average basket size
    - Product category diversity
    - Household demographics if available
    """
    # Convert date strings to datetime objects
    tx['Date'] = pd.to_datetime(tx['Date'])
    
    # Get the latest date in the dataset to calculate recency
    latest_date = tx['Date'].max()
    training_cutoff_date = latest_date - pd.Timedelta(days=90)
    
    # Split transactions into training and evaluation periods
    tx_train = tx[tx['Date'] <= training_cutoff_date]
    
    # Group training transactions by household
    customer_stats = tx_train.groupby('Hshd_num').agg(
        last_purchase_date=('Date', 'max'),
        first_purchase_date=('Date', 'min'),
        purchase_count=('Basket_num', 'nunique'),
        total_spend=('Spend', 'sum'),
        total_units=('Units', 'sum'),
        avg_basket_value=('Spend', lambda x: x.sum() / len(set(tx_train.loc[x.index]['Basket_num']))),
        unique_products=('Product_num', 'nunique'),
        store_region=('Store_region', lambda x: x.mode()[0] if not x.empty else None)
    ).reset_index()
    
    # Calculate tenure (days between first and last purchase)
    customer_stats['tenure'] = (customer_stats['last_purchase_date'] - customer_stats['first_purchase_date']).dt.days
    
    # Add a small value to tenure to avoid division by zero
    customer_stats['tenure_safe'] = customer_stats['tenure'].clip(lower=1)  # Minimum tenure of 1 day
    
    # Calculate average purchase frequency (purchases per 30 days of tenure)
    customer_stats['purchase_frequency'] = customer_stats['purchase_count'] / (customer_stats['tenure_safe'] / 30)
    
    # Merge with household demographics if available
    if not households.empty:
        customer_stats = customer_stats.merge(households, on='Hshd_num', how='left')
    
    # Calculate spend per visit
    customer_stats['spend_per_visit'] = customer_stats['total_spend'] / customer_stats['purchase_count'].clip(lower=1)
    
    # Calculate product diversity ratio
    customer_stats['product_diversity_ratio'] = customer_stats['unique_products'] / customer_stats['total_units'].clip(lower=1)
    
    # Define churn based on post-training period activity
    # Find households that made purchases in training period
    training_households = set(tx_train['Hshd_num'].unique())
    
    # Find households that made purchases in evaluation period (after training cutoff)
    tx_eval = tx[tx['Date'] > training_cutoff_date]
    active_households = set(tx_eval['Hshd_num'].unique())
    
    # A customer has churned if they made purchases in training period but not in evaluation period
    churned_households = training_households - active_households
    
    # Mark churned households
    customer_stats['churned'] = customer_stats['Hshd_num'].isin(churned_households).astype(int)
    
    churn_rate = customer_stats['churned'].mean()
    print(f"Churn rate: {churn_rate:.2%}")
    
    return customer_stats

def train_churn_model(customer_stats):
    """
    Train a Gradient Boosting model to predict customer churn.
    """
    # Select features and target
    # Explicitly exclude recency and last_purchase_date to avoid data leakage
    numeric_cols = ['purchase_count', 'total_spend', 'total_units', 
                    'avg_basket_value', 'unique_products', 'tenure', 
                    'purchase_frequency', 'spend_per_visit', 'product_diversity_ratio']
    
    # Add demographic features if available
    if 'Age_range' in customer_stats.columns:
        # One-hot encode categorical variables
        demo_features = pd.get_dummies(customer_stats[['Age_range', 'Marital_status', 'Income_range', 'Homeowner_desc', 'Hshd_composition', 'Hshd_size', 'Children']], 
                                      drop_first=True)
        
        # Combine numeric and demographic features
        X = pd.concat([customer_stats[numeric_cols], demo_features], axis=1)
    else:
        X = customer_stats[numeric_cols]
    
    y = customer_stats['churned']
    
    # Check for and handle any NaN or infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # Print out columns with NaN values for debugging
    na_cols = X.columns[X.isna().any()].tolist()
    if na_cols:
        print(f"Columns with NaN values: {na_cols}")
        # Fill NaN values with column means
        X = X.fillna(X.mean())
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train a gradient boosting classifier
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=10,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance.head(10))
    
    # Return the model, scaler, and feature columns for future predictions
    return model, scaler, X.columns.tolist()

def main():
    print("Loading data...")
    tx, households = load_data()
    
    print("Preparing features...")
    customer_stats = prepare_features(tx, households)
    
    print("Training churn prediction model...")
    model, scaler, feature_cols = train_churn_model(customer_stats)
    
    print("Saving model, scaler, and feature columns to churn_prediction_model.pkl...")
    joblib.dump((model, scaler, feature_cols), "churn_prediction_model.pkl")
    
    # Save customer stats for dashboard visualization
    customer_stats.to_csv("customer_churn_stats.csv", index=False)
    
    print("Done.")

if __name__ == "__main__":
    main()
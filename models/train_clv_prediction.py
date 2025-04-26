import pandas as pd
import numpy as np
import joblib
import pyodbc
import config
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_data():
    """
    Load Transactions and Households tables from Azure SQL via pyodbc.
    """
    conn = pyodbc.connect(config.ODBC_CONN_STR)
    
    # Get transactions data
    tx = pd.read_sql("SELECT * FROM Transactions;", conn)
    
    # Get household data
    households = pd.read_sql("SELECT * FROM Households;", conn)
    
    # Get product data for category information
    products = pd.read_sql("SELECT * FROM Products;", conn)
    
    conn.close()
    return tx, households, products

def prepare_features(tx, households, products):
    """
    Prepare features for CLV prediction:
    - Frequency: Number of transactions in the period
    - Monetary: Total spend amount
    - Recency: Time since last purchase
    - Tenure: Time since first purchase
    - Product category preferences
    - Household demographics if available
    """
    # Convert date strings to datetime objects
    tx['Date'] = pd.to_datetime(tx['Date'])
    
    # Get the maximum date in the dataset
    latest_date = tx['Date'].max()
    
    # Define training period (e.g., first 75% of data)
    # and prediction period (remaining 25%)
    training_cutoff_date = latest_date - pd.Timedelta(days=90)
    
    # Split transactions into training and evaluation periods
    tx_train = tx[tx['Date'] <= training_cutoff_date]
    tx_eval = tx[tx['Date'] > training_cutoff_date]
    
    # Merge transactions with products to get category information
    tx_with_products = tx_train.merge(products, on='Product_num', how='left')
    
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
    
    # Calculate purchase frequency (purchases per month)
    customer_stats['purchase_frequency'] = customer_stats['purchase_count'] / (customer_stats['tenure_safe'] / 30)
    
    # Calculate recency (days since last purchase)
    customer_stats['recency'] = (training_cutoff_date - customer_stats['last_purchase_date']).dt.days
    
    # Calculate average spend per day of tenure
    customer_stats['spend_per_day'] = customer_stats['total_spend'] / customer_stats['tenure_safe']
    
    # Calculate spend per visit
    customer_stats['spend_per_visit'] = customer_stats['total_spend'] / customer_stats['purchase_count'].clip(lower=1)
    
    # Calculate product diversity ratio
    customer_stats['product_diversity'] = customer_stats['unique_products'] / customer_stats['total_units'].clip(lower=1)
    
    # Calculate category preferences
    category_preferences = tx_with_products.groupby(['Hshd_num', 'Department']).agg(
        category_spend=('Spend', 'sum')
    ).reset_index()
    
    # Pivot to get category spending as columns
    category_pivot = category_preferences.pivot_table(
        index='Hshd_num', 
        columns='Department', 
        values='category_spend',
        fill_value=0
    )
    
    # Rename columns to avoid spaces and special characters
    category_pivot.columns = [f'dept_{col.lower().replace(" ", "_")}' for col in category_pivot.columns]
    
    # Merge with customer stats
    customer_stats = customer_stats.merge(category_pivot, on='Hshd_num', how='left')
    
    # Merge with household demographics if available
    if not households.empty:
        customer_stats = customer_stats.merge(households, on='Hshd_num', how='left')
    
    # Calculate target: future spend (CLV) based on spend in evaluation period
    future_spend = tx_eval.groupby('Hshd_num').agg(
        future_spend=('Spend', 'sum')
    ).reset_index()
    
    # Merge the future spend with customer stats
    customer_stats = customer_stats.merge(future_spend, on='Hshd_num', how='left')
    
    # Fill missing future spend with 0 (for customers who didn't purchase in evaluation period)
    customer_stats['future_spend'] = customer_stats['future_spend'].fillna(0)
    
    # Project CLV for 12 months based on 90-day future spend
    # Multiply by 4 to annualize (assuming 90 days is approximately 1 quarter)
    customer_stats['annual_clv'] = customer_stats['future_spend'] * 4
    
    # Additional CLV adjustment based on historical metrics
    # For customers with longer tenure, we have more confidence in our predictions
    confidence_factor = np.minimum(customer_stats['tenure_safe'] / 365, 1)  # Cap at 1 year
    historical_monthly_spend = customer_stats['total_spend'] / (customer_stats['tenure_safe'] / 30)
    
    # Blend historical and future projections based on confidence
    customer_stats['blended_monthly_spend'] = (
        (confidence_factor * historical_monthly_spend) + 
        ((1 - confidence_factor) * (customer_stats['future_spend'] / 3))  # Future spend per month
    )
    
    # Final CLV estimate for 1 year
    customer_stats['clv_12_month'] = customer_stats['blended_monthly_spend'] * 12
    
    # Print some statistics for debugging
    print(f"Average 12-month CLV: ${customer_stats['clv_12_month'].mean():.2f}")
    print(f"Median 12-month CLV: ${customer_stats['clv_12_month'].median():.2f}")
    print(f"Max 12-month CLV: ${customer_stats['clv_12_month'].max():.2f}")
    
    return customer_stats

def train_clv_model(customer_stats):
    """
    Train a Random Forest model to predict customer lifetime value.
    """
    # Select features and target
    numeric_cols = ['purchase_count', 'total_spend', 'total_units', 
                    'avg_basket_value', 'unique_products', 'tenure', 
                    'purchase_frequency', 'recency', 'spend_per_day',
                    'spend_per_visit', 'product_diversity']
    
    # Add department spending columns to numeric features
    dept_cols = [col for col in customer_stats.columns if col.startswith('dept_')]
    numeric_cols.extend(dept_cols)
    
    # Add demographic features if available
    demo_features = []
    if 'Age_range' in customer_stats.columns:
        # One-hot encode categorical variables
        cat_columns = ['Age_range', 'Marital_status', 'Income_range', 
                       'Homeowner_desc', 'Hshd_composition', 'Hshd_size', 'Children']
        available_cat_cols = [col for col in cat_columns if col in customer_stats.columns]
        
        if available_cat_cols:
            demo_features = pd.get_dummies(customer_stats[available_cat_cols], drop_first=True)
    
    # Combine numeric and demographic features
    if not demo_features.empty:
        X = pd.concat([customer_stats[numeric_cols], demo_features], axis=1)
    else:
        X = customer_stats[numeric_cols]
    
    # Target is 12-month CLV
    y = customer_stats['clv_12_month']
    
    # Check for and handle any NaN or infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # Print out columns with NaN values for debugging
    na_cols = X.columns[X.isna().any()].tolist()
    if na_cols:
        print(f"Columns with NaN values: {na_cols}")
        # Fill NaN values with column means
        X = X.fillna(X.mean())
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train a Random Forest regressor
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    
    # Calculate regression metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\nModel Evaluation:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R² Score: {r2:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Feature Importance:")
    print(feature_importance.head(10))
    
    # For high-value customer segmentation, let's divide customers into tiers
    customer_stats['predicted_clv'] = model.predict(scaler.transform(X))
    
    # Define CLV tiers (quartiles)
    customer_stats['clv_tier'] = pd.qcut(
        customer_stats['predicted_clv'], 
        q=4, 
        labels=['Bronze', 'Silver', 'Gold', 'Platinum']
    )
    
    # Count customers in each tier
    tier_counts = customer_stats['clv_tier'].value_counts()
    print("\nCustomer Tiers:")
    for tier, count in tier_counts.items():
        tier_avg = customer_stats[customer_stats['clv_tier'] == tier]['predicted_clv'].mean()
        print(f"{tier}: {count} customers, Avg CLV: ${tier_avg:.2f}")
    
    # Return the model, scaler, and feature columns for future predictions
    return model, scaler, X.columns.tolist()

def main():
    print("Loading data...")
    tx, households, products = load_data()
    
    print("Preparing features...")
    customer_stats = prepare_features(tx, households, products)
    
    print("Training CLV prediction model...")
    model, scaler, feature_cols = train_clv_model(customer_stats)
    
    print("Saving model, scaler, and feature columns to clv_prediction_model.pkl...")
    joblib.dump((model, scaler, feature_cols), "clv_prediction_model.pkl")
    
    # Save customer stats with CLV predictions for dashboard visualization
    customer_stats.to_csv("customer_clv_stats.csv", index=False)
    
    print("Done.")

if __name__ == "__main__":
    main()
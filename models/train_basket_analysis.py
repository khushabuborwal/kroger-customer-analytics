import pandas as pd
import numpy as np
import joblib
import pyodbc
import config
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import itertools
import os

def load_data():
    """
    Load Transactions and Products tables from Azure SQL via pyodbc.
    """
    conn = pyodbc.connect(config.ODBC_CONN_STR)
    tx = pd.read_sql("SELECT * FROM Transactions;", conn)
    products = pd.read_sql("SELECT * FROM Products;", conn)
    conn.close()
    return tx, products

def create_basket_pairs(tx, products):
    print("Creating basket pairs...")
    tx_products = tx.merge(products, on='Product_num', how='left')
    # product_id as string for pairing
    tx_products['product_id'] = tx_products['Product_num'].astype(str)
    basket_groups = tx_products.groupby(['Hshd_num', 'Basket_num'])
    pair_counts = {}
    for (_, _), group in basket_groups:
        prods = group['product_id'].unique()
        if len(prods) < 2:
            continue
        for p1, p2 in itertools.combinations(prods, 2):
            pair = tuple(sorted([p1, p2]))
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
    pairs_df = pd.DataFrame([
        {'product1': p[0], 'product2': p[1], 'count': c}
        for p, c in pair_counts.items()
    ]).sort_values('count', ascending=False)
    return pairs_df, tx_products

def calculate_product_features(tx_products):
    print("Calculating product features...")
    cols = ['product_id']
    if 'Department' in tx_products.columns:
        cols.append('Department')
    if 'Commodity' in tx_products.columns:
        cols.append('Commodity')
    dept_comm = tx_products[cols].drop_duplicates()
    stats = tx_products.groupby('product_id').agg(
        avg_spend=('Spend','mean'),
        total_spend=('Spend','sum'),
        purchase_count=('Basket_num','count'),
        avg_units=('Units','mean'),
        total_units=('Units','sum')
    ).reset_index()
    stats = stats.merge(dept_comm, on='product_id', how='left')
    total_baskets = tx_products[['Hshd_num','Basket_num']].drop_duplicates().shape[0]
    stats['purchase_frequency'] = stats['purchase_count'] / total_baskets
    if 'Loyalty_flag' in tx_products.columns:
        loy = tx_products.groupby('product_id')['Loyalty_flag'].mean().reset_index()
        loy.rename(columns={'Loyalty_flag':'loyalty_ratio'}, inplace=True)
        stats = stats.merge(loy, on='product_id', how='left')
    print(f"Product stats columns: {stats.columns.tolist()}")
    return stats

def prepare_model_data(pairs_df, product_stats):
    print("Preparing model data...")

    # Copy and cast pair IDs to int
    md = pairs_df.copy()
    md['product1'] = md['product1'].astype(int)
    md['product2'] = md['product2'].astype(int)

    # Ensure stats IDs are int
    product_stats['product_id'] = product_stats['product_id'].astype(int)

    # Merge product1 features
    p1 = product_stats.rename(columns={
        col: f"p1_{col}" for col in product_stats.columns if col != "product_id"
    })
    md = md.merge(p1, left_on="product1", right_on="product_id", how="left")\
           .drop("product_id", axis=1)

    # Merge product2 features
    p2 = product_stats.rename(columns={
        col: f"p2_{col}" for col in product_stats.columns if col != "product_id"
    })
    md = md.merge(p2, left_on="product2", right_on="product_id", how="left")\
           .drop("product_id", axis=1)

    # Combined features
    md["total_frequency"] = md["p1_purchase_frequency"] + md["p2_purchase_frequency"]
    md["price_similarity"] = 1 - abs(md["p1_avg_spend"] - md["p2_avg_spend"]) / (
        md["p1_avg_spend"] + md["p2_avg_spend"] + 1e-3
    )
    md["same_department"] = (
        (md["p1_Department"] == md["p2_Department"]).astype(int)
        if "p1_Department" in md.columns and "p2_Department" in md.columns
        else 0
    )
    md["normalized_count"] = md["count"] / md["count"].max()

    # One-hot encode any categorical
    cats = [
        c for c in md.columns
        if (c.startswith("p1_") or c.startswith("p2_"))
        and any(x in c for x in ["Department", "Commodity"])
    ]
    if cats:
        md = pd.get_dummies(md, columns=cats, dummy_na=True)

    print(f"Model data columns: {md.columns.tolist()}")
    return md


def train_basket_model(model_data):
    print("Training basket analysis model...")
    num_cols = [c for c in model_data.columns if c.startswith(('p1_','p2_')) 
                or c in ['total_frequency','price_similarity','same_department']]
    num_cols = [c for c in num_cols if model_data[c].dtype in [np.int64, np.float64]]
    print(f"Using {len(num_cols)} numeric features: {num_cols[:5]}...")
    X = model_data[num_cols].replace([np.inf,-np.inf],np.nan).fillna(0)
    y = model_data['normalized_count']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)
    scaler = StandardScaler()
    Xs_train, Xs_test = scaler.fit_transform(X_train), scaler.transform(X_test)
    model = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(Xs_train, y_train)
    y_pred = model.predict(Xs_test)
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test,y_pred)):.4f}, R2: {r2_score(y_test,y_pred):.4f}")
    fi = pd.DataFrame({'feature':num_cols,'importance':model.feature_importances_})
    print("Feature Importance:\n", fi.sort_values('importance', ascending=False).head(10))
    lookup = {(row['product1'],row['product2']):idx for idx,row in model_data.iterrows()}
    lookup.update({(row['product2'],row['product1']):idx for idx,row in model_data.iterrows()})
    return model, scaler, num_cols, lookup, model_data

def prepare_recommendations_data(pairs_df, product_stats, product_info):
    print("Preparing recommendations data...")
    # cast ids
    product_stats['product_id'] = product_stats['product_id'].astype(int)
    top_pairs = pairs_df.head(200).copy()
    top_pairs['product1'] = top_pairs['product1'].astype(int)
    top_pairs['product2'] = top_pairs['product2'].astype(int)
    # build products_df
    pids = pd.Series(list(set(top_pairs['product1']).union(top_pairs['product2'])))
    products_df = pd.DataFrame({'product_id':pids.astype(int)})
    products_df = products_df.merge(product_stats, on='product_id', how='left')
    mapping = (product_info[['Product_num','Department','Commodity']] 
               .drop_duplicates() 
               .rename(columns={'Product_num':'product_id','Department':'dept','Commodity':'comm'}))
    products_df = products_df.merge(mapping, on='product_id', how='left')
    # details
    product_details = {}
    for _,row in products_df.iterrows():
        product_details[int(row['product_id'])] = {
            'product_num': int(row['product_id']),
            'frequency': row.get('purchase_frequency',0),
            'department': row.get('dept',''),
            'commodity':  row.get('comm','')
        }
    # recommendations
    product_recommendations = {}
    for _,row in top_pairs.iterrows():
        p1,p2,cnt = int(row['product1']), int(row['product2']), int(row['count'])
        product_recommendations.setdefault(p1,[]).append((p2,cnt))
        product_recommendations.setdefault(p2,[]).append((p1,cnt))
    for k in product_recommendations:
        product_recommendations[k].sort(key=lambda x: x[1], reverse=True)
    return product_details, product_recommendations

def save_model_and_data(model, scaler, feature_cols, product_details, product_recommendations):
    print("Saving model and data...")
    joblib.dump((model,scaler,feature_cols),"basket_analysis_model.pkl")
    joblib.dump(product_details,      "product_details.pkl")
    joblib.dump(product_recommendations,"product_recommendations.pkl")
    # write top pairs csv
    top_pairs_list = []
    for p1, recs in product_recommendations.items():
        for p2,cnt in recs[:5]:
            if p1 in product_details and p2 in product_details:
                pd1 = product_details[p1]
                pd2 = product_details[p2]
                top_pairs_list.append({
                    'product1_num': pd1['product_num'],
                    'product1_dept': pd1['department'],
                    'product1_comm':pd1['commodity'],
                    'product2_num': pd2['product_num'],
                    'product2_dept': pd2['department'],
                    'product2_comm':pd2['commodity'],
                    'co_occurrence_count': cnt
                })
    pd.DataFrame(top_pairs_list).drop_duplicates().to_csv("top_product_pairs.csv",index=False)
    print("Saved model and data files.")

def main():
    print("Starting basket analysis model training...")
    tx,products = load_data()
    pairs_df, tx_products = create_basket_pairs(tx, products)
    product_stats = calculate_product_features(tx_products)
    model_data = prepare_model_data(pairs_df, product_stats)
    model, scaler, feature_cols, lookup, model_data = train_basket_model(model_data)
    product_details, product_recommendations = prepare_recommendations_data(pairs_df, product_stats, products)
    save_model_and_data(model, scaler, feature_cols, product_details, product_recommendations)
    print("Basket analysis model training complete.")

if __name__ == "__main__":
    main()

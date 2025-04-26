from flask import Flask, render_template, request, redirect, flash, session, url_for, jsonify
import pyodbc
import pandas as pd
import json
import os
import numpy as np
import joblib
import sklearn


# load the trained churn model, scaler, and feature list
model, scaler, feature_cols = joblib.load("saved_models/churn_prediction_model.pkl")

# preload the customer stats CSV that your training script exported
customer_stats_df = pd.read_csv("stats/customer_churn_stats.csv")

# load the trained CLV model, scaler, and feature list
clv_model, clv_scaler, clv_feature_cols = joblib.load("saved_models/clv_prediction_model.pkl")

# preload the customer stats CSV that your CLV training script exported
customer_clv_df = pd.read_csv("stats/customer_clv_stats.csv")

# load your trained basket‐analysis model, scaler, and feature list
basket_model, basket_scaler, basket_features = joblib.load("saved_models/basket_analysis_model.pkl")

# load the product metadata & recommendation mappings
product_details       = joblib.load("saved_models/product_details.pkl")
product_recommendations = joblib.load("saved_models/product_recommendations.pkl")

# load the precomputed top‐pairs table
top_pairs_df = pd.read_csv("stats/top_product_pairs.csv")


app = Flask(__name__)
app.secret_key = 'cloud_final_project'  # Needed for sessions and flash messages

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Azure SQL Database connection settings
server = 'retail-app-server.database.windows.net'
database = 'kroger-database'
username = 'Khushbu-Afreen'
password = 'KAProject1!'
driver = '{ODBC Driver 17 for SQL Server}'  # Ensure you have this installed

def get_db_connection():
    conn = pyodbc.connect(
        'DRIVER=' + driver + 
        ';SERVER=' + server + 
        ';PORT=1433;DATABASE=' + database + 
        ';UID=' + username + 
        ';PWD=' + password
    )
    return conn

# Simulated user database
users = {}

# Login Page
@app.route('/')
def home():
    return render_template('login.html')  # Changed to 'login.html'

# Registration Page
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']

        if username in users:
            flash("Username already exists!", "danger")
            return redirect('/register')
        
        users[username] = {'password': password, 'email': email}
        flash("Registered successfully. Please login.", "success")
        # TODO: Save user data securely
        return redirect('/')  # Redirect to login after successful registration
    return render_template('register.html')

# Login Submission Handling
@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']

    user = users.get(username)

    if not user:  # Check if the user is not registered
            flash("User not registered. Please register first.", "danger")
            return redirect(url_for('register'))  # Redirect to the register page
    
    if user and user['password'] == password:
        session['username'] = username
        return redirect('/home')
    else:
        flash("Invalid credentials. Please try again.", "danger")
        return redirect('/')
    
@app.route('/home')
def home_page():
    if 'username' not in session:
        return redirect('/')
    
    filtered_data = []
    hshd_num = 10

    conn = get_db_connection()

    query = """
            SELECT T.Hshd_num, T.Basket_num, T.date, P.Product_num,
                   P.Department, P.Commodity, T.Spend, T.Units, T.Store_region, T.Week_num, T.Year
            FROM Transactions T
            JOIN Products P ON T.Product_num = P.Product_num
            WHERE T.Hshd_num = ?
            ORDER BY T.Hshd_num, T.Basket_num, T.date, P.Product_num
        """

    df = pd.read_sql(query, conn, params=(hshd_num,))
    filtered_data = df.to_dict(orient='records')
    conn.close()

    return render_template('home.html', username=session['username'], data=filtered_data, hshd_num=hshd_num)

# Search Household number
@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        hshd_num = int(request.form['hshd_num'])
        conn = get_db_connection()

        query = """
            SELECT T.Hshd_num, T.Basket_num, T.date, P.Product_num,
                   P.Department, P.Commodity, T.Spend, T.Units, T.Store_region, T.Week_num, T.Year
            FROM Transactions T
            JOIN Products P ON T.Product_num = P.Product_num
            WHERE T.Hshd_num = ?
            ORDER BY T.Hshd_num, T.Basket_num, T.date, P.Product_num
        """

        df = pd.read_sql(query, conn, params=(hshd_num,))
        filtered_data = df.to_dict(orient='records')
        
        conn.close()

        return render_template('home.html', hshd_num=hshd_num, data=filtered_data)

    return render_template('home.html', data=None)

@app.route('/dashboard')
def dashboard():
    # Check if user is logged in
    if 'username' not in session:
        flash("Please log in first.")
        return redirect(url_for('login'))

    # Establish database connection
    conn = get_db_connection()

    # Query 1: Demographics and Engagement
    query_demo = """
        SELECT 
            T.Store_region, 
            H.Age_range, 
            AVG(T.Spend) AS avg_spend, 
            AVG(T.Units) AS avg_units
        FROM 
            Transactions T
        JOIN 
            Households H ON T.Hshd_num = H.Hshd_num
        GROUP BY 
            T.Store_region, H.Age_range
    """
    df_demo = pd.read_sql(query_demo, conn)

    # Query 2: Engagement Over Time
    query_time = """
        SELECT Year, Month(Date) as Month, SUM(Spend) as total_spend
        FROM Transactions
        GROUP BY Year, Month(Date)
        ORDER BY Year, Month(Date)
    """
    df_time = pd.read_sql(query_time, conn)

    # Query 3: Basket Analysis
    query_basket = """
        SELECT TOP 5 P.Commodity, COUNT(*) AS count FROM Transactions T
        JOIN Products P ON T.Product_num = P.Product_num
        GROUP BY P.Commodity
        ORDER BY COUNT(*) DESC
    """
    df_basket = pd.read_sql(query_basket, conn)

    # Query 4: Seasonal Trends
    query_season = """
        SELECT Month(Date) as Month, SUM(Spend) as total_spend
        FROM Transactions
        GROUP BY Month(Date)
        ORDER BY Month(Date)
    """
    df_season = pd.read_sql(query_season, conn)

    # Query 5: Brand Preferences
    query_brand = """
        SELECT H.Loyalty_flag, COUNT(*) AS count
        FROM Transactions T
        JOIN Households H ON T.Hshd_num = H.Hshd_num
        GROUP BY H.Loyalty_flag
    """
    df_brand = pd.read_sql(query_brand, conn)

    # Query 6: Units Sold vs. Total Spend for each Commodity
    query_spend = """
        SELECT p.Commodity, SUM(t.Units) AS total_units, SUM(t.Spend) AS total_spend
        FROM  TRANSACTIONS t
        JOIN PRODUCTS p ON t.Product_num = p.Product_num
        GROUP BY p.Commodity
        ORDER BY total_spend DESC
    """

    df_spend = pd.read_sql(query_spend, conn)

    # Close the database connection
    conn.close()

    # Convert data to JSON for the frontend
    data = {
        "demographics": df_demo.to_dict(orient='records'),
        "engagement_over_time": df_time.to_dict(orient='records'),
        "basket_analysis": df_basket.to_dict(orient='records'),
        "seasonal_trends": df_season.to_dict(orient='records'),
        "brand_preferences": df_brand.to_dict(orient='records'),
        "total_spend": df_spend.to_dict(orient='records'),
    }

    # Render the dashboard template with data
    return render_template('dashboard.html', data=json.dumps(data))

# UploD CSV data
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    # Check if user is logged in
    if 'username' not in session:
        flash("Please log in first.")
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        transactions_file = request.files.get('transactions')
        households_file = request.files.get('households')
        products_file = request.files.get('products')

        if not transactions_file or not households_file or not products_file:
            flash("All three files are required!", "danger")
            return redirect('/upload')

        # Save files locally
        transactions_path = os.path.join(UPLOAD_FOLDER, 'transactions.csv')
        households_path = os.path.join(UPLOAD_FOLDER, 'households.csv')
        products_path = os.path.join(UPLOAD_FOLDER, 'products.csv')

        transactions_file.save(transactions_path)
        households_file.save(households_path)
        products_file.save(products_path)

        # Read with pandas
        transactions_df = pd.read_csv(transactions_path)
        households_df = pd.read_csv(households_path)
        products_df = pd.read_csv(products_path)

        # Standardize column names
        for df in [transactions_df, households_df, products_df]:
            df.columns = df.columns.str.strip().str.upper()

        # Insert into database
        conn = get_db_connection()
        cursor = conn.cursor()

        # Clear existing data first (optional)
        # cursor.execute("DELETE FROM Transactions")
        # cursor.execute("DELETE FROM Households")
        # cursor.execute("DELETE FROM Products")
        # conn.commit()

        # Fill missing values for Households
        households_df['HH_SIZE'] = pd.to_numeric(households_df['HH_SIZE'], errors='coerce').fillna(0).astype(int)
        households_df['CHILDREN'] = pd.to_numeric(households_df['CHILDREN'], errors='coerce').fillna(0).astype(int)
        # Insert Households
        for index, row in households_df.iterrows():
            cursor.execute("""
                INSERT INTO Households (Hshd_num, Loyalty_flag, Age_range, Marital_status, 
                                        Income_range, Homeowner_desc, Hshd_composition, 
                                        Hshd_size, Children)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, row['HSHD_NUM'], row['L'], row['AGE_RANGE'], row['MARITAL'], 
                 row['INCOME_RANGE'], row['HOMEOWNER'], row['HSHD_COMPOSITION'],
                 row['HH_SIZE'], row['CHILDREN'])
        conn.commit()

        # Fill missing values for products
        products_df['PRODUCT_NUM'] = pd.to_numeric(products_df['PRODUCT_NUM'], errors='coerce').fillna(0).astype(int)

        # Insert Products
        for index, row in products_df.iterrows():
            try:
                cursor.execute("""
                    INSERT INTO Products (Product_num, Department, Commodity, Brand_type, Natural_organic_flag)
                    VALUES (?, ?, ?, ?, ?)
                """, row['PRODUCT_NUM'], row['DEPARTMENT'], row['COMMODITY'],
                    row['BRAND_TY'], row['NATURAL_ORGANIC_FLAG'])
            except Exception as e:
                print(f"Skipping product row {index} due to error: {e}")
        conn.commit()

        # Fill missing numeric values for Transactions
        transactions_df['SPEND'] = pd.to_numeric(transactions_df['SPEND'], errors='coerce').fillna(0)
        transactions_df['UNITS'] = pd.to_numeric(transactions_df['UNITS'], errors='coerce').fillna(0)
        transactions_df['WEEK_NUM'] = pd.to_numeric(transactions_df['WEEK_NUM'], errors='coerce').fillna(0)
        transactions_df['YEAR'] = pd.to_numeric(transactions_df['YEAR'], errors='coerce').fillna(0)
        # Insert Transactions
        for index, row in transactions_df.iterrows():
            try:
                cursor.execute("""
                    INSERT INTO Transactions (Basket_num, Hshd_num, Date, Product_num, Spend, Units, Store_region, Week_num, Year)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, row['BASKET_NUM'], row['HSHD_NUM'], row['PURCHASE_'], row['PRODUCT_NUM'],
                    row['SPEND'], row['UNITS'], row['STORE_R'], row['WEEK_NUM'], row['YEAR'])
            except Exception as e:
                print(f"Skipping transaction row {index} due to error: {e}")
        conn.commit()

        # NEW: Count how many rows inserted
        households_count = len(households_df)
        products_count = len(products_df)
        transactions_count = len(transactions_df)

        print("households_count---------", households_count)
        print("products_count---------", products_count)
        print("transactions_count---------", transactions_count)

        conn.close()

        # Flash success with counts
        flash(f"Upload successful! {households_count} Households, {products_count} Products, {transactions_count} Transactions inserted.", "success")
        return redirect('/home')
    
    # Render the dashboard template with data
    return render_template('upload.html', data=[])


# Churn Prediction
@app.route('/churn-prediction')
def churn_prediction():
    df = customer_stats_df.copy()
    # --- align features exactly to feature_cols, filling any new/missing cols with 0 ---
    X = pd.DataFrame(index=df.index, columns=feature_cols)
    for col in feature_cols:
        if col in df.columns:
            X[col] = df[col].fillna(0)
        else:
            X[col] = 0
    # --- scale and predict ---
    X_scaled    = scaler.transform(X)
    df['churn_prob'] = model.predict_proba(X_scaled)[:, 1]

    # sort descending by risk
    preds = df[['Hshd_num', 'churn_prob']].sort_values('churn_prob', ascending=False)
    predictions = preds.to_dict(orient='records')

    # feature-to-churn correlation (only for columns that actually exist in df)
    corr_df = df[['churned'] + [c for c in feature_cols if c in df.columns]]
    corr    = corr_df.corr()['churned'].drop('churned')
    correlations = corr.to_dict()

    return render_template(
        'churn-prediction.html',
        predictions=predictions,
        correlations=correlations
    )


# CLV Prediction
@app.route('/clv-prediction')
def clv_prediction():
    df = customer_clv_df.copy()

    # === align features exactly ===
    X = pd.DataFrame(index=df.index, columns=clv_feature_cols)
    for col in clv_feature_cols:
        X[col] = df[col].fillna(0) if col in df.columns else 0

    # scale & predict
    X_scaled    = clv_scaler.transform(X)
    df['clv_pred'] = clv_model.predict(X_scaled)

    # top‐n predictions
    preds = df[['Hshd_num', 'clv_pred']]\
            .sort_values('clv_pred', ascending=False)\
            .to_dict(orient='records')

    # === CLV distribution histogram (10 bins) ===
    counts, bins = np.histogram(df['clv_pred'], bins=10)
    # create human‐readable bin labels
    dist_labels = [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(counts))]
    dist_counts = counts.tolist()

    # === feature importance (top 10) ===
    fi_df = pd.DataFrame({
        'feature': clv_feature_cols,
        'importance': clv_model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    fi_names  = fi_df['feature'].tolist()
    fi_values = fi_df['importance'].tolist()

    return render_template(
        'clv-prediction.html',
        predictions=preds,
        dist_labels=dist_labels,
        dist_counts=dist_counts,
        fi_names=fi_names,
        fi_values=fi_values
    )

# Serve the single-page basket analysis UI
@app.route("/basket-analysis")
def basket_analysis():
    return render_template("basket-analysis.html")


# Return all products for the multi-select
@app.route("/api/products")
def api_products():
    # product_details keys are product_id strings
    out = []
    for pid, det in product_details.items():
        out.append({
            "id": int(pid),             # convert back to int
            "department": det["department"],
            "commodity": det["commodity"]
        })
    return jsonify(out)


# Recommend cross-sells using your model + lookup maps
@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    basket = request.get_json().get("products", [])
    # aggregate counts across all selected products
    rec_acc = {}
    for pid in basket:
        # product_recommendations keys are ints, so use pid directly
        for other_pid, cnt in product_recommendations.get(pid, []):
            rec_acc[other_pid] = rec_acc.get(other_pid, 0) + cnt

    if not rec_acc:
        return jsonify([])

    # normalize to [0,1]
    max_cnt = max(rec_acc.values())
    recs = []
    for pid2, cnt in rec_acc.items():
        det = product_details.get(pid2, {})
        recs.append({
          "department": det.get("department", "Unknown"),
          "commodity": det.get("commodity", "Unknown"),
          "score": cnt / max_cnt
        })
    # sort by descending score
    recs.sort(key=lambda x: x["score"], reverse=True)
    return jsonify(recs)


# Return the overall top-pairs from your saved CSV
@app.route("/api/top-pairs")
def api_top_pairs():
    # top_pairs_df has columns: product1_num, product2_num, co_occurrence_count, product1_dept, product1_comm, product2_dept, product2_comm
    out = []
    for row in top_pairs_df.itertuples():
        pair_name = f"{row.product1_dept} – {row.product1_comm} & {row.product2_dept} – {row.product2_comm}"
        out.append({"pair": pair_name, "count": int(row.co_occurrence_count)})
    return jsonify(out)

if __name__ == "__main__":
    app.run(debug=True)

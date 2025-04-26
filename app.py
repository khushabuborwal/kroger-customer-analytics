from flask import Flask, render_template, request, redirect, flash, session, url_for
import pyodbc
import pandas as pd
import json
app = Flask(__name__)
app.secret_key = 'cloud_final_project'  # Needed for sessions and flash messages

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

if __name__ == "__main__":
    app.run(debug=True)

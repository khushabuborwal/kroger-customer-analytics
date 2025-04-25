from flask import Flask, render_template, request, redirect, flash, session, url_for

app = Flask(__name__)
app.secret_key = 'cloud_final_project'  # Needed for sessions and flash messages

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
    
    # Sample data – replace with DB query in real app
    sample_data = [
        {'Hshd_num': 10, 'Basket_num': 201, 'Purchase_Date': '2018-08-19', 'Product_num': '153300', 'Department': 'FOOD', 'Commodity': 'GROCERY STAPLE', 'Spend': 7.29, 'Units': 1, 'Store_region': 'EAST', 'Week_num': 33, 'Year': 2018},
        {'Hshd_num': 10, 'Basket_num': 201, 'Purchase_Date': '2018-08-19', 'Product_num': '248739', 'Department': 'PHARMA', 'Commodity': 'MEDICATION', 'Spend': 2.99, 'Units': 1, 'Store_region': 'EAST', 'Week_num': 33, 'Year': 2018},
        {'Hshd_num': 10, 'Basket_num': 201, 'Purchase_Date': '2018-08-19', 'Product_num': '153300', 'Department': 'FOOD', 'Commodity': 'GROCERY STAPLE', 'Spend': 7.29, 'Units': 1, 'Store_region': 'EAST', 'Week_num': 33, 'Year': 2018},
        {'Hshd_num': 10, 'Basket_num': 201, 'Purchase_Date': '2018-08-19', 'Product_num': '248739', 'Department': 'PHARMA', 'Commodity': 'MEDICATION', 'Spend': 2.99, 'Units': 1, 'Store_region': 'EAST', 'Week_num': 33, 'Year': 2018},
        {'Hshd_num': 10, 'Basket_num': 201, 'Purchase_Date': '2018-08-19', 'Product_num': '153300', 'Department': 'FOOD', 'Commodity': 'GROCERY STAPLE', 'Spend': 7.29, 'Units': 1, 'Store_region': 'EAST', 'Week_num': 33, 'Year': 2018},
        {'Hshd_num': 10, 'Basket_num': 201, 'Purchase_Date': '2018-08-19', 'Product_num': '248739', 'Department': 'PHARMA', 'Commodity': 'MEDICATION', 'Spend': 2.99, 'Units': 1, 'Store_region': 'EAST', 'Week_num': 33, 'Year': 2018},
        {'Hshd_num': 10, 'Basket_num': 201, 'Purchase_Date': '2018-08-19', 'Product_num': '153300', 'Department': 'FOOD', 'Commodity': 'GROCERY STAPLE', 'Spend': 7.29, 'Units': 1, 'Store_region': 'EAST', 'Week_num': 33, 'Year': 2018},
        {'Hshd_num': 10, 'Basket_num': 201, 'Purchase_Date': '2018-08-19', 'Product_num': '248739', 'Department': 'PHARMA', 'Commodity': 'MEDICATION', 'Spend': 2.99, 'Units': 1, 'Store_region': 'EAST', 'Week_num': 33, 'Year': 2018},
        {'Hshd_num': 10, 'Basket_num': 201, 'Purchase_Date': '2018-08-19', 'Product_num': '153300', 'Department': 'FOOD', 'Commodity': 'GROCERY STAPLE', 'Spend': 7.29, 'Units': 1, 'Store_region': 'EAST', 'Week_num': 33, 'Year': 2018},
        {'Hshd_num': 10, 'Basket_num': 201, 'Purchase_Date': '2018-08-19', 'Product_num': '248739', 'Department': 'PHARMA', 'Commodity': 'MEDICATION', 'Spend': 2.99, 'Units': 1, 'Store_region': 'EAST', 'Week_num': 33, 'Year': 2018},
        # Add more rows here...
    ]

    return render_template('home.html', username=session['username'], data=sample_data)

# Search Household number
@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        hshd_num = int(request.form['hshd_num'])

        # Replace this with a DB query later
        all_data = [
            {'Hshd_num': 10, 'Basket_num': 201, 'Purchase_Date': '2018-08-19', 'Product_num': '153300', 'Department': 'FOOD', 'Commodity': 'GROCERY STAPLE', 'Spend': 7.29, 'Units': 1, 'Store_region': 'EAST', 'Week_num': 33, 'Year': 2018},
            {'Hshd_num': 10, 'Basket_num': 201, 'Purchase_Date': '2018-08-19', 'Product_num': '248739', 'Department': 'PHARMA', 'Commodity': 'MEDICATION', 'Spend': 2.99, 'Units': 1, 'Store_region': 'EAST', 'Week_num': 33, 'Year': 2018},
            {'Hshd_num': 11, 'Basket_num': 205, 'Purchase_Date': '2018-09-01', 'Product_num': '189009', 'Department': 'FOOD', 'Commodity': 'BAKERY', 'Spend': 3.50, 'Units': 1, 'Store_region': 'EAST', 'Week_num': 34, 'Year': 2018},
        ]

        filtered = [row for row in all_data if row['Hshd_num'] == hshd_num]

        return render_template('home.html', hshd_num=hshd_num, data=filtered)

    return render_template('home.html', data=None)

@app.route('/dashboard')
def dashboard():
    # Check if user is logged in
    if 'username' not in session:
        flash("Please log in first.")
        return redirect(url_for('login'))

    # # Establish database connection
    # conn = get_db_connection()

    # # Query 1: Demographics and Engagement
    # query_demo = """
    #     SELECT Store_region, Age_range, AVG(Spend) as avg_spend, AVG(Units) as avg_units
    #     FROM TransformedData
    #     GROUP BY Store_region, Age_range
    # """
    # df_demo = pd.read_sql(query_demo, conn)

    # # Query 2: Engagement Over Time
    # query_time = """
    #     SELECT Year, Month(Purchase_Date) as Month, SUM(Spend) as total_spend
    #     FROM TransformedData
    #     GROUP BY Year, Month(Purchase_Date)
    #     ORDER BY Year, Month(Purchase_Date)
    # """
    # df_time = pd.read_sql(query_time, conn)

    # # Query 3: Basket Analysis
    # query_basket = """
    #     SELECT TOP 5 Commodity, COUNT(*) as count
    #     FROM TransformedData
    #     GROUP BY Commodity
    #     ORDER BY COUNT(*) DESC
    # """
    # df_basket = pd.read_sql(query_basket, conn)

    # # Query 4: Seasonal Trends
    # query_season = """
    #     SELECT Month(Purchase_Date) as Month, SUM(Spend) as total_spend
    #     FROM TransformedData
    #     GROUP BY Month(Purchase_Date)
    #     ORDER BY Month(Purchase_Date)
    # """
    # df_season = pd.read_sql(query_season, conn)

    # # Query 5: Brand Preferences
    # query_brand = """
    #     SELECT Loyalty_flag, COUNT(*) as count
    #     FROM TransformedData
    #     GROUP BY Loyalty_flag
    # """
    # df_brand = pd.read_sql(query_brand, conn)

    # # Close the database connection
    # conn.close()

    # # Convert data to JSON for the frontend
    # data = {
    #     "demographics": df_demo.to_dict(orient='records'),
    #     "engagement_over_time": df_time.to_dict(orient='records'),
    #     "basket_analysis": df_basket.to_dict(orient='records'),
    #     "seasonal_trends": df_season.to_dict(orient='records'),
    #     "brand_preferences": df_brand.to_dict(orient='records'),
    # }

    # Render the dashboard template with data
    return render_template('dashboard.html', data=[])

if __name__ == "__main__":
    app.run(debug=True)

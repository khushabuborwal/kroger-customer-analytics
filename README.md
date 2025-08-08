# 🛒 Retail Analytics Platform – Azure Cloud Computing

[![Azure](https://img.shields.io/badge/Powered%20by-Microsoft%20Azure-0089D6?logo=microsoft-azure)](https://azure.microsoft.com)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python)](https://www.python.org/)


**Live Demo:** [Retail Analytics Web App](https://retailapp-hnhjfeetebeee5b4.eastus2-01.azurewebsites.net/)  

A cloud-based retail analytics platform that predicts **Customer Lifetime Value (CLV)**, identifies **cross-selling opportunities**, and detects **customers at risk of churn**.  
Built entirely on **Microsoft Azure** with **end-to-end data pipelines**, **machine learning models**, and an **interactive dashboard**.

---

## 📑 Table of Contents
- [✨ Features](#-features)
- [🛠 Tech Stack](#-tech-stack)
- [⚙️ Architecture](#️-architecture)
- [📊 Machine Learning Models](#-machine-learning-models)
- [📂 Project Workflow](#-project-workflow)
- [📸 Screenshots](#-screenshots)
- [📈 Business Impact](#-business-impact)

---

## ✨ Features

### **1. Customer Lifetime Value (CLV) Prediction**
- **Model:** Random Forest Regression  
- Predicts **12-month CLV** from:
  - RFM metrics (Recency, Frequency, Monetary)
  - Tenure, spend patterns, product diversity, demographics
- **Segmentation:** Bronze → Silver → Gold → Platinum
- **Business Actions:**
  - Platinum: VIP perks, personalized outreach
  - Gold: Targeted cross-sells
  - Silver/Bronze: Frequency & basket size incentives

---

### **2. Basket Analysis for Cross-Selling**
- **Model:** Gradient Boosting  
- Identifies high-value product pairings for promotions  
- **Top Examples:**
  - 🥛 Milk + 🍌 Bananas
  - 🍞 Bread + 🧀 Cheese
  - 🍚 Rice + 🥦 Vegetables  
- **Applications:**
  - In-checkout recommendations
  - Bundled discounts
  - Store layout optimization

---

### **3. Customer Churn Prediction**
- **Model:** Gradient Boosting  
- Flags at-risk customers and reveals churn drivers  
- **Key Insights:**
  - Tenure (–0.85) → longterm customers churn less
  - High product diversity (+0.78) → higher churn risk  
- **Retention Tactics:**
  - Welcome campaigns for new sign-ups
  - “We miss you” offers for lapsers
  - Frequency-based reminders

---

## 🛠 Tech Stack

**Cloud Platform:**
- Microsoft Azure

**Azure Services:**
- Azure SQL Database
- Azure Blob Storage
- Azure Data Factory
- Azure App Service

**Machine Learning:**
- Python (Pandas, Scikit-learn)
- Random Forest Regression
- Gradient Boosting

**Web Application:**
- Python / Flask
- HTML, CSS, JavaScript

---

## ⚙️ Architecture

CSV Data → Azure Blob Storage  
         → Azure Data Factory → Azure SQL Database  
         → ML Model Training & Prediction  
         → Flask Web App on Azure App Service  
         → Interactive Dashboards  
         
📊 Machine Learning Models  
| Task             | Model             | Reason for Choice                                  |
| ---------------- | ----------------- | -------------------------------------------------- |
| CLV Prediction   | Random Forest     | Handles non-linear features, robust, interpretable |
| Basket Analysis  | Gradient Boosting | Captures complex co-purchase patterns              |
| Churn Prediction | Gradient Boosting | Works well with skewed churn data                  |


## 📂 Project Workflow

1. Data Ingestion
- Store raw CSV files in Azure Blob Storage
- Use Azure Data Factory to load into SQL tables

2. Data Processing
- Clean, transform, and feature engineer data
- Prepare datasets for ML models

3. Model Training
- Train models for CLV, basket analysis, churn prediction
- Evaluate accuracy and interpret feature importance

4. Web Deployment
- Flask app hosted on Azure App Service
- Interactive dashboards with data

## 📸 Screenshots

- Login Page -  
![Login Page UI](https://github.com/khushabuborwal/kroger-customer-analytics/blob/main/UI-Screeshots/login.png "Login Page UI")

- Home Page -  
![Home Page UI](https://github.com/khushabuborwal/kroger-customer-analytics/blob/main/UI-Screeshots/Homepage.png "Home Page UI")

- Dashboard -  
![Dashboard UI](https://github.com/khushabuborwal/kroger-customer-analytics/blob/main/UI-Screeshots/Dashboard.png "Dashboard UI")

- Upload Retail Data -  
![Upload_Data UI](https://github.com/khushabuborwal/kroger-customer-analytics/blob/main/UI-Screeshots/Upload-data.png "Upload_Data UI")

- Customer Lifetime Value (CLV) -  
![CLV UI](https://github.com/khushabuborwal/kroger-customer-analytics/blob/main/UI-Screeshots/CLV.png "CLV UI")

- Basket Analysis -  
![Basket_Analysis UI](https://github.com/khushabuborwal/kroger-customer-analytics/blob/main/UI-Screeshots/Basket.png "Basket_Analysis UI")

- Customer Churn -  
![Customer_Churn UI](https://github.com/khushabuborwal/kroger-customer-analytics/blob/main/UI-Screeshots/Churn.png "Customer_Churn UI")  


## 📈 Business Impact
🎯 Focused Marketing – Prioritize Platinum customers for retention  
💰 Revenue Boost – Strategic cross-selling increases basket value  
🔄 Reduced Churn – Targeted campaigns to retain at-risk customers  


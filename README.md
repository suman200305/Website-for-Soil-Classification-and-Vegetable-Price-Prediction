# Website-for-Soil-Classification-and-Vegetable-Price-Prediction
🌾 Smart.A: Vegetable Price Prediction & Soil Classification Platform

Smart.A is an AI-powered web application designed to assist farmers and agricultural stakeholders by providing:​

Vegetable Price Forecasting using time-series models.
Soil Type Classification through deep learning techniques.
Soil Characteristics Analysis and Crop Recommendations based on classified soil types.​

📌 Table of Contents
Overview

Features

Tech Stack

Project Structure

Installation

Usage

Screenshots

Contributors

License

📖 Overview
Smart.A integrates machine learning and deep learning models into a user-friendly web interface to provide:​

Vegetable Price Prediction: Forecasts future vegetable prices based on historical data, aiding farmers in market planning.

Soil Type Classification: Identifies soil types from uploaded images, providing insights into soil properties.

Crop Recommendation: Suggests suitable crops based on soil characteristics, enhancing agricultural productivity.​

🚀 Features
1. Vegetable Price Prediction
Utilizes SARIMA models for accurate time-series forecasting.

Interactive UI to select state, district, market, vegetable, and start date.

Displays predicted prices in tabular format.​
GitHub

2. Soil Type Classification
Employs a Convolutional Neural Network (CNN) trained on diverse soil images.

Allows users to upload soil images for classification.

Provides confidence scores and detailed soil characteristics.​

3. Crop Recommendation
Based on classified soil types, recommends optimal crops.

Offers insights into soil properties and suitable agricultural practices.​

🛠 Tech Stack
Frontend: HTML, CSS, JavaScript

Backend: Python, Flask

Machine Learning: SARIMA (Statsmodels) for price prediction

Deep Learning: TensorFlow/Keras for soil classification

Data Handling: Pandas, NumPy

Visualization: Matplotlib, Seaborn​

📁 Project Structure
cpp
Copy
Edit

├── app.py
├── static/
│   ├── css/
│   ├── js/
│   └── images/
├── templates/
│   ├── index.html
│   ├── service1.html
│   └── service2.html
├── models/
│   ├── soil_model.h5
│   └── price_model.pkl
├── data/
│   └── vegetables_prices_additional_data.csv
├── requirements.txt
└── README.md
⚙ Installation
Clone the repository:

bash
Copy
Edit
git clone https://github.com/yourusername/SmartA.git
cd SmartA
Create a virtual environment:

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the application:

bash
Copy
Edit
python app.py
Access the application:

Navigate to http://localhost:5000 in your web browser.

💻 Usage
Home Page: Provides an overview and navigation to services.

Vegetable Price Prediction: Select relevant parameters to forecast prices.

Soil Type Classification: Upload a soil image to determine its type and receive crop recommendations.​

📸 Screenshots
Vegetable Price Prediction

Soil Type Classification

👥 Contributors
Abhik Sarkar
Tushar Paul

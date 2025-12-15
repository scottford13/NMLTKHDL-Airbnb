# NMLTKHDL-Airbnb
A mini-project for the course Introduction to Programming for Data Science analyzes the New York City Airbnb Open Data to predict listing price based on various features such as location, room type and availability. 

## **Table of Contents**
1. Introduction
2. Dataset
3. Methodology
4. Installation & Setup
5. Usage
6. Results
7. Project Structure
8. Challenges & Solutions
9. Future Improvements
10. Contributors
11. Contact
12. License

## **1. Introduction**
- Pricing an Airbnb rental is a complex task. Hosts want to maximize revenue without scaring away guests, while guests look for the best value. The core problem addressed in this project is to analyze and model the relationship between listing characteristics and rental prices, with the goal of identifying key factors that influence pricing and evaluating how well prices can be predicted using available features. This project's goal is to build a Linear Regression model from scratch using only NumPy (without Scikit-learn or Pandas) to predict the price of a listing.

## **2. Dataset**
- Source: https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data/data
- Description: The dataset includes listing activity and metrics in NYC, NY for 2019.
- The dataset has `48895` rows and `16` columns.
- Key Features:
    - `neighbourhood_group`: Location (e.g., Manhattan, Brooklyn).
    - `room_type`: Entire home, Private room, Shared room.
    - `minimum_nights`: Minimum stay required.
    - `number_of_reviews`, `reviews_per_month`: Listing activity.
    - `availability_365`: Number of days available per year.
- Target: `price` (USD per night).

## **3. Methodology**
### **a. Data Preprocessing (Pure NumPy)**
- The data processing pipeline was implemented without Pandas, utilizing NumPy for efficient array manipulation:
    - Cleaning: Handled missing values in reviews_per_month by filling with 0.
    - Outlier Removal: Removed listings with price <= 0.
    - Log Transformation: Applied np.log1p to the target variable price to handle the heavy right-skew of monetary data.
    - Encoding: Implemented One-Hot Encoding manually using NumPy indexing to convert categorical variables (neighbourhood_group, room_type) into numerical vectors.
    - Scaling: Applied Z-score normalization (Standardization) to numerical features.
### **b. Model: Linear Regression**
- The model was built from scratch using the Normal Equation to find the optimal parameters $\theta$ that minimize the cost function (Mean Squared Error).
- Mathematical Formula:
$$\theta = (X^T X)^{-1} X^T y
$$Where:$X$ is the matrix of input features (with a bias term $x_0 = 1$ added).$y$ is the vector of target values (log-prices).$\theta$ is the vector of coefficients.The prediction is calculated using vectorization:

## **4. Installation & Setup**
## **5. Usage**
## **6. Results**
## **7. Project Structure**
## **8. Challenges & Solutions**
## **9. Future Improvements**
## **10. Contributors**
## **11. Contact**
## **12. License**
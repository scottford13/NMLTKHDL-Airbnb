# LTKHDL-Airbnb
A mini-project for the course Programming for Data Science analyzes the New York City Airbnb Open Data to predict listing price based on various features such as location, room type and availability. 

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
Pricing an Airbnb rental is a complex task. Hosts want to maximize revenue without scaring away guests, while guests look for the best value. The core problem addressed in this project is to analyze and model the relationship between listing characteristics and rental prices, with the goal of identifying key factors that influence pricing and evaluating how well prices can be predicted using available features. This project's goal is to build a Linear Regression model from scratch using only NumPy (without Scikit-learn or Pandas) to predict the price of a listing.

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
The model was built from scratch using the Normal Equation to find the optimal parameters $\theta$ that minimize the cost function (Mean Squared Error).
Mathematical Formula:
$$\theta = (X^T X)^{-1} X^T y$$
Where:
    - $X$ is the matrix of input features (with a bias term $x_0 = 1$ added).
    - $y$ is the vector of target values (log-prices).
    - $\theta$ is the vector of coefficients.
The prediction is calculated using vectorization:
$$\hat{y} = X \cdot \theta$$

## **4. Installation & Setup**
### **a. Clone the repository**
git clone https://github.com/scottford13/LTKHDL-Airbnb.git  
cd LTKHDL-Airbnb
### **b. Install dependencies**
pip install -r requirements.txt  
*Dependencies include: numpy, matplotlib, seaborn, jupyter.*

## **5. Usage**
**Run the notebooks in the following order to reproduce the analysis**
### **a. Exploration**
jupyter notebook notebooks/01_data_exploration.ipynb  

*Visualizes distributions and correlations*
### **b. Preprocessing**
jupyter notebook notebooks/02_preprocessing.ipynb  

*Cleans data, encodes features, and saves train_data.csv / test_data.csv.*
### **c. Modeling**
jupyter notebook notebooks/03_modeling.ipynb  

*Trains the NumPy Linear Regression model and outputs metrics.*

## **6. Results**
**Metrics**
The model was evaluated on the test set using Root Mean Squared Error (RMSE) and R-squared ($R^2$):
- RMSE (after inverse log transformation): 49.47 USD
- $R^2$ Score: 0.4777  

**Visualizations**
- Actual vs. Predicted: The model captures the general trend well, though it tends to underpredict high-end luxury listings (due to the "long tail" of prices).
- Residuals: The residuals follow a bell-curve distribution, indicating the model errors are random and unbiased.
- Feature Importance: room_type_Entire home/apt and neighbourhood_group_Manhattan were found to be the strongest drivers of higher prices.

## **7. Project Structure**
```
project-name/
├── README.md               # Project documentation    
├── requirements.txt        # Python dependencies    
├── data/  
│   ├── raw/                # Original AB_NYC_2019.csv  
│   └── processed/          # Processed train/test CSV files  
├── notebooks/  
│   ├── 01_data_exploration.ipynb  
│   ├── 02_preprocessing.ipynb  
│   └── 03_modeling.ipynb  
└── src/  
    ├── __init__.py  
    ├── data_processing.py  # Helper functions for loading/cleaning  
    └── models.py           
    └── visualization.py      
```

## **8. Challenges & Solutions**
| Challenge                 | Solution                                                                 |
|---------------------------|--------------------------------------------------------------------------|
| No Pandas for CSV         | Used `csv` module and `np.genfromtxt` to parse mixed data types.         |
| Manual One-Hot Encoding   | Used `np.unique` and boolean indexing to create binary feature columns.  |
| Matrix Invertibility      | Used `np.linalg.pinv` instead of `np.linalg.inv` to handle singularity.  |


## **9. Future Improvements**
- Regularization: Implement Ridge Regression (L2) to penalize large coefficients and reduce overfitting.
- Optimization: Implement Gradient Descent for better scalability on larger datasets where matrix inversion is too costly.
- Advanced Features: Engineer new features based on text reviews (NLP) or distance to subway stations.

## **10. Contributors**
- Student Name: Pham Anh Khoa
- Student ID: 22127198
- Dataset's Owner: Dgomonov
- For any questions, please contact pakhoa22@clc.fitus.edu.vn

## **11. License**
- This project is licensed under the MIT License.
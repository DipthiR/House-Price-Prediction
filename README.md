# House-Price-Prediction
# 🏠 House Price Prediction

A simple Machine Learning project that predicts house prices using linear regression. The model is trained on housing data and uses features like the number of bedrooms, bathrooms, square footage, year built, and more.

---

## 📌 Project Overview

This project uses a CSV dataset containing housing features to build a regression model that predicts house prices. It includes:

- Data cleaning and preprocessing
- Encoding categorical variables
- Linear Regression model training
- Evaluation using MAE and R² score
- Visual comparison of predicted vs actual prices
- Custom input prediction

---

## 📁 Dataset

The dataset should be a CSV file with at least these columns:

- `price`
- `bedrooms`
- `bathrooms`
- `sqft_living`
- `sqft_lot`
- `floors`
- `waterfront`
- `view`
- `condition`
- `sqft_above`
- `sqft_basement`
- `yr_built`
- `yr_renovated`
- `street`
- `city`
- `statezip`
- `country`

You can use the [`House prediction.csv`](https://www.kaggle.com/) or any custom dataset with similar columns.

---

## 🛠 How to Run

1. Clone the repository or download the code.
2. Make sure you have Python 3.x installed.
3. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
Place your dataset (e.g., House prediction.csv) in the project folder.

Run the script:

python house_price_prediction.py
## 📊 Output
Evaluation Metrics

Mean Absolute Error (MAE)

R² Score

Scatter Plot

Actual vs Predicted Prices

Custom Prediction

Predict the price of a manually input custom house.

## 🧠 Model Used
Linear Regression from scikit-learn

## 🧪 Example Custom House

custom_values = [3, 2, 2000, 4000, 1, 0, 0, 3, 1800, 200, 1995, 0, 1200, 500, 98001, 1]
Feel free to change these values to see how the price prediction responds.

## 🚀 Future Improvements
Use advanced regression models (Random Forest, XGBoost)

Hyperparameter tuning

Web interface with Flask or Streamlit

Handle more categorical features with One-Hot Encoding

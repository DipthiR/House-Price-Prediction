import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("C:\\Users\\dipthi\\Downloads\\House prediction.csv")

# Clean column names: strip spaces and convert to lowercase
df.columns = df.columns.str.strip().str.lower()

# Check available columns
print("Columns in dataset:", df.columns.tolist())

# Optional: preview the data
print(df.head())

# Drop 'date' column (or extract features like year/month separately if needed)
if 'date' in df.columns:
    df = df.drop(columns=['date'])

# Encode categorical variables
categorical_cols = ['street', 'city', 'statezip', 'country']
for col in categorical_cols:
    if col in df.columns:
        df[col] = LabelEncoder().fit_transform(df[col])

# Define features (only use columns that exist in your dataset)
features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
            'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement',
            'yr_built', 'yr_renovated', 'street', 'city', 'statezip', 'country']

# Filter only columns that exist in your dataset
features = [col for col in features if col in df.columns]

# Check if 'price' column exists
if 'price' not in df.columns:
    raise ValueError("The dataset does not contain a 'price' column.")

X = df[features]
y = df['price']

# Handle missing values
X = X.fillna(X.mean())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("\nModel Performance:")
print("Mean Absolute Error:", round(mean_absolute_error(y_test, y_pred), 2))
print("R2 Score:", round(r2_score(y_test, y_pred), 4))

# Plot results
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.show()

# Custom house prediction (adjust values according to your dataset)
custom_values = [3, 2, 2000, 4000, 1, 0, 0, 3, 1800, 200, 1995, 0, 1200, 500, 98001, 1]
custom_input = pd.DataFrame([custom_values[:len(features)]], columns=features)

predicted_price = model.predict(custom_input)
print("\nPredicted Price for the custom house: $", round(predicted_price[0], 2))

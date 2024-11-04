# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import zscore

# Load the dataset
url = "uber.csv"  # Replace with the path to your dataset file
df = pd.read_csv(url)

# Task 1: Data Preprocessing
# Checking for null values and data types
print(df.info())
print(df.describe())

# Convert pickup and drop-off datetime columns to datetime type if needed
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

# Extract useful features from pickup_datetime (e.g., day, hour, etc.)
df['pickup_day'] = df['pickup_datetime'].dt.day
df['pickup_hour'] = df['pickup_datetime'].dt.hour
df['pickup_dayofweek'] = df['pickup_datetime'].dt.dayofweek

# Drop unnecessary columns if any
df.drop(['pickup_datetime', 'key'], axis=1, inplace=True)

# Remove rows with missing values
df.dropna(inplace=True)

# Task 2: Identify Outliers
# Using Z-score to identify outliers in the 'fare_amount' column
z_scores = zscore(df['fare_amount'])
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3)  # Keeping entries with z-score < 3
df = df[filtered_entries]

# Task 3: Check Correlation
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# Task 4: Implement Models
# Define features and target variable
X = df[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'pickup_day', 'pickup_hour', 'pickup_dayofweek']]
y = df['fare_amount']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_lr = linear_model.predict(X_test)

# Random Forest Regression Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Task 5: Model Evaluation
# Define a function to evaluate the model
def evaluate_model(y_test, y_pred):
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return r2, rmse

# Evaluate Linear Regression
r2_lr, rmse_lr = evaluate_model(y_test, y_pred_lr)
print("Linear Regression - R2:", r2_lr)
print("Linear Regression - RMSE:", rmse_lr)

# Evaluate Random Forest Regression
r2_rf, rmse_rf = evaluate_model(y_test, y_pred_rf)
print("Random Forest Regression - R2:", r2_rf)
print("Random Forest Regression - RMSE:", rmse_rf)

# Comparison of Model Performance
print("\nModel Performance Comparison:")
print("Linear Regression - R2:", r2_lr, ", RMSE:", rmse_lr)
print("Random Forest Regression - R2:", r2_rf, ", RMSE:", rmse_rf)

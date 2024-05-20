# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Step 1: Load the dataset
# Assuming the dataset is in a CSV file named 'house_prices.csv'
df = pd.read_csv('house_prices.csv')

# Step 2: Data Exploration and Visualization
# Display the first few rows of the dataframe
print(df.head())

# Display summary information about the dataframe
print(df.info())

# Display basic statistics of the dataframe
print(df.describe())

# Visualize the distribution of house prices
sns.histplot(df['Price'], kde=True)
plt.title('Distribution of House Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# Visualize relationships between features and the target variable
# Assuming 'Size', 'Bedrooms', 'Bathrooms' are relevant features
sns.pairplot(df, x_vars=['Size', 'Bedrooms', 'Bathrooms'], y_vars='Price', height=5, aspect=0.75, kind='reg')
plt.show()

# Step 3: Data Preparation
# Handle missing values (if any)
df.fillna(df.mean(), inplace=True)

# Feature selection (example features: 'Size', 'Bedrooms', 'Bathrooms')
features = ['Size', 'Bedrooms', 'Bathrooms']
X = df[features]
y = df['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling (standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Build and Train the Model
# Using Linear Regression as the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Step 5: Model Evaluation
# Predict house prices on the test set
y_pred = model.predict(X_test_scaled)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R-squared: {r2}')

# Step 6: Visualize Predictions vs Actual Values
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted House Prices')
plt.show()

# Save the model for future use
joblib.dump(model, 'house_price_model.pkl')
print("Model saved as 'house_price_model.pkl'")

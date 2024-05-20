# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Step 1: Load the dataset
# Assuming the dataset is in a CSV file named 'heart.csv'
df = pd.read_csv('heart.csv')

# Display the first few rows of the dataframe
print(df.head())

# Display summary information about the dataframe
print(df.info())

# Display basic statistics of the dataframe
print(df.describe())

# Step 2: Data Exploration and Visualization
# Visualize the distribution of target variable (presence of heart disease)
sns.countplot(x='target', data=df)
plt.title('Distribution of Heart Disease')
plt.xlabel('Heart Disease')
plt.ylabel('Count')
plt.show()

# Visualize relationships between features and the target variable
# Example features: 'age', 'sex', 'cp' (chest pain type), 'thalach' (maximum heart rate achieved)
sns.pairplot(df, vars=['age', 'sex', 'cp', 'thalach'], hue='target', markers=["o", "s"], diag_kind="kde")
plt.show()

# Step 3: Data Preparation
# Define feature columns and target variable
X = df.drop(columns='target')
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Build and Train the Model
# Using Logistic Regression as the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Step 5: Model Evaluation
# Predict the target on the test set
y_pred = model.predict(X_test_scaled)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(cm)
print('Classification Report:')
print(report)

# Step 6: Visualize Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Save the model
joblib.dump(model, 'heart_disease_model.pkl')
print("Model saved as 'heart_disease_model.pkl'")

# Import Libraries
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Dataset (saved as credit.csv)
df = pd.read_csv("credit.csv")

# Select relevant features
df = df[['Pclass', 'Sex', 'Age', 'Fare', 'Survived']]

# Handle missing values
df['Age'] = df['Age'].fillna(df['Age'].mean())

# Encode categorical data (Sex)
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

# Define Features and Target
X = df.drop('Survived', axis=1)
y = df['Survived']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model (Logistic Regression)
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Example Prediction
sample = [[3, 1, 25, 300]]   # (Pclass, Sex, Age, Fare)
sample_scaled = scaler.transform(sample)
prediction = model.predict(sample_scaled)

print("\nSample Prediction:", "Low Risk" if prediction[0]==1 else "High Risk")
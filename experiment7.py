import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_csv("stock.csv")

# Fix column names automatically
df.columns = df.columns.str.lower()

# Convert date
df['date'] = pd.to_datetime(df['date'])

# Sort
df = df.sort_values('date')

# Select columns
df = df[['date', 'close']]

# Create prediction
df['prediction'] = df['close'].shift(-1)
df = df.dropna()

# Features & target
X = df[['close']].values
y = df['prediction'].values

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))

plt.plot(y_test, label="Actual")
plt.plot(y_pred, label="Predicted")
plt.legend()
plt.show()
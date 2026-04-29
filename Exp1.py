import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import OneHotEncoder

# Load dataset
dataset = pd.read_excel("HousePricePrediction.xlsx")
print(dataset.head())

# Identify columns
object_cols = dataset.select_dtypes(include=['object']).columns
int_cols = dataset.select_dtypes(include=['int64']).columns
float_cols = dataset.select_dtypes(include=['float64']).columns

print("Categorical variables:", len(object_cols))
print("Integer variables:", len(int_cols))
print("Float variables:", len(float_cols))

# Correlation heatmap
numerical_dataset = dataset.select_dtypes(include=['int64', 'float64'])
plt.figure(figsize=(12, 6))
sns.heatmap(numerical_dataset.corr(), cmap='BrBG', annot=True)
plt.show()

# Drop Id safely
if 'Id' in dataset.columns:
    dataset.drop(['Id'], axis=1, inplace=True)

# Handle missing values
dataset['SalePrice'] = dataset['SalePrice'].fillna(dataset['SalePrice'].mean())
dataset = dataset.dropna()

# One-hot encoding
dataset = pd.get_dummies(dataset, drop_first=True)

# Split data
X = dataset.drop('SalePrice', axis=1)
Y = dataset['SalePrice']

X_train, X_valid, Y_train, Y_valid = train_test_split(
    X, Y, test_size=0.2, random_state=0
)

# Train model
model_LR = LinearRegression()
model_LR.fit(X_train, Y_train)

# Predict
Y_pred = model_LR.predict(X_valid)

# Evaluate
print("MAPE:", mean_absolute_percentage_error(Y_valid, Y_pred))

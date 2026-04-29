# Import Libraries
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

# Load Dataset
data = load_breast_cancer()
X = data.data
y = data.target   # 0 = malignant, 1 = benign

# Convert target for anomaly detection
# Let's treat malignant (0) as anomaly
y_true = np.where(y == 0, 1, 0)

# Train Isolation Forest
model = IsolationForest(contamination=0.1, random_state=42)
model.fit(X)

# Predict
y_pred = model.predict(X)

# Convert output (-1 anomaly → 1, normal → 0)
y_pred = np.where(y_pred == -1, 1, 0)

# Evaluation
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
print("\nClassification Report:\n", classification_report(y_true, y_pred))
# Import Libraries
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load Image Dataset
digits = load_digits()

X = digits.data       # image features
y = digits.target     # labels (0–9 classes)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model (SVM)
model = SVC(kernel='rbf')
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Show sample image
plt.imshow(digits.images[0], cmap='gray')
plt.title(f"Predicted: {model.predict([digits.data[0]])[0]}")
plt.show()
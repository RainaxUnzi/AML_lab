# Import Libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.metrics import accuracy_score

# Load MNIST Dataset
(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

# Normalize data (0-255 → 0-1)
X_train = X_train / 255.0
X_test = X_test / 255.0

# Build Neural Network Model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),   # convert 2D → 1D
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 digits (0–9)
])

# Compile Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train Model
model.fit(X_train, y_train, epochs=5)

# Evaluate Model
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_acc)

# Prediction Example
predictions = model.predict(X_test)
print("Predicted Digit:", np.argmax(predictions[0]))
print("Actual Digit:", y_test[0])
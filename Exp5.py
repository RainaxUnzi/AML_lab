# Import Libraries
import numpy as np
import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Create Dataset
data = {
    "text": [
        "I love this product", "This is amazing", "Very happy with service",
        "Excellent quality", "Superb experience", "I like it a lot",
        "Worst product ever", "I hate this", "Very bad experience",
        "Not good at all", "Terrible service", "Waste of money",
        "Highly satisfied", "Best purchase", "Loved it",
        "Disappointed", "Poor quality", "Not worth it"
    ],
    "label": [1,1,1,1,1,1, 0,0,0,0,0,0, 1,1,1, 0,0,0]
}

df = pd.DataFrame(data)

# Step 2: Text Preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

df['clean_text'] = df['text'].apply(clean_text)

# Step 3: Features and Target
X = df['clean_text']
y = df['label']

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Step 5: Convert Text to Numbers (TF-IDF)
vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Step 6: Train Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 7: Prediction
y_pred = model.predict(X_test)

# Step 8: Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 9: Test Custom Input
sample = ["This product is really good"]
sample_clean = [clean_text(sample[0])]
sample_vec = vectorizer.transform(sample_clean)
prediction = model.predict(sample_vec)

print("\nCustom Input Prediction:",
      "Positive" if prediction[0] == 1 else "Negative")
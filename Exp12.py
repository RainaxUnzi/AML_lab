# Import Libraries
import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Create Dataset (Real + Fake News)
data = {
    "text": [
        "Government announces new economic policy",
        "Scientists discover new vaccine",
        "Stock market reaches all time high",
        "Fake news spreads misinformation online",
        "Celebrity caught in fake scandal",
        "Click here to win money instantly",
        "Breaking: new education reforms announced",
        "Doctors warn about fake health tips",
        "Win a free iPhone now",
        "Important government notice released"
    ],
    "label": [1,1,1,0,0,0,1,0,0,1]   # 1 = Real, 0 = Fake
}

df = pd.DataFrame(data)

# Step 2: Text Cleaning
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
    X, y, test_size=0.2, random_state=42
)

# Step 5: TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Step 6: Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 7: Prediction
y_pred = model.predict(X_test)

# Step 8: Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 9: Custom Input
sample = ["Breaking: Government launches new scheme"]
sample_clean = [clean_text(sample[0])]
sample_vec = vectorizer.transform(sample_clean)

prediction = model.predict(sample_vec)

print("\nPrediction:", "Real News" if prediction[0]==1 else "Fake News")
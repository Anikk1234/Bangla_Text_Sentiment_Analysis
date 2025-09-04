import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from stopwords import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

nltk.download('punkt')

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', str(text))
    tokens = word_tokenize(text)
    stop_words = set(stopwords)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('cleaned_dataset.csv', encoding='utf-8-sig')

# Preprocess the text
print("Preprocessing text...")
df.dropna(subset=['Text'], inplace=True)
df['processed_comment'] = df['Text'].apply(preprocess_text)

# Map sentiment tags to numerical labels
sentiment_map = {'Negative': 0, 'Positive': 1}
df['Label'] = df['Tag'].map(sentiment_map)

# Drop rows where mapping resulted in NaN (if any unmapped tags exist)
df.dropna(subset=['Label'], inplace=True)
y = df['Label'].astype(int)

# TF-IDF Vectorization
print("Vectorizing text...")
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['processed_comment'])

# Split the data
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
print("Training model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate the model
print("Evaluating model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)

# Save the accuracy and classification report
with open("accuracy.txt", "w") as f:
    f.write(str(accuracy))
with open("classification_report.txt", "w") as f:
    f.write(report)

# Save vocabulary to file
print("Saving vocabulary to file...")
with open("vocabulary.txt", "w", encoding="utf-8") as f:
    for word in vectorizer.get_feature_names_out():
        f.write(word + '\n')


# Save the model and vectorizer
print("Saving model and vectorizer...")
script_dir = os.path.dirname(os.path.abspath(__file__))
joblib.dump(model, os.path.join(script_dir, 'sentiment_model.joblib'))
joblib.dump(vectorizer, os.path.join(script_dir, 'tfidf_vectorizer.joblib'))

print("Training complete.")
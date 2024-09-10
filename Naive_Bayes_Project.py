
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
import pickle
import streamlit as st

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load dataset
df = pd.read_csv(r'encoded_spam.csv')

# Drop the last 3 columns
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)

# Renaming the columns
df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)

# Encode the target labels
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])

# Remove duplicates
df = df.drop_duplicates(keep='first')

# Initialize the Porter Stemmer
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    y = [ps.stem(i) for i in y]
    return " ".join(y)

# Apply text transformation
df['transformed_text'] = df['text'].apply(transform_text)

# Initialize TfidfVectorizer
tfidf = TfidfVectorizer(max_features=3000)
X_sparse = tfidf.fit_transform(df['transformed_text'])  # Sparse matrix
X = X_sparse.toarray()  # Convert to dense numpy array
y = df['target'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Initialize and train the Gaussian Naive Bayes model
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred1 = gnb.predict(X_test)
print("Accuracy - ", accuracy_score(y_test, y_pred1))
print("Confusion Matrix - \n", confusion_matrix(y_test, y_pred1))
print("Precision - ", precision_score(y_test, y_pred1))

# Save the TF-IDF vectorizer and the model
pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
pickle.dump(gnb, open('model.pkl', 'wb'))
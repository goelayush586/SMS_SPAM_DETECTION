#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
import seaborn as sns

# Load the dataset
df = pd.read_csv(r'encoded_spam.csv')

# Drop last 3 columns
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)

# Rename the columns
df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)

# Encode the target variable
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])

# Text preprocessing function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)

# Apply text preprocessing
ps = PorterStemmer()
df['transformed_text'] = df['text'].apply(transform_text)

# TF-IDF vectorization
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['transformed_text']).toarray()
y = df['target'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Initialize Decision Tree model
dt = DecisionTreeClassifier()

# Fit Decision Tree model on training data
dt.fit(X_train, y_train)

# Predictions using Decision Tree model
y_pred_dt = dt.predict(X_test)

# Accuracy, confusion matrix, and precision score for Decision Tree
accuracy_dt = accuracy_score(y_test, y_pred_dt)
confusion_matrix_dt = confusion_matrix(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt)

# Plotting accuracy and precision for Decision Tree
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.bar(['Decision Tree'], [accuracy_dt], color='skyblue')
plt.title('Accuracy Scores')
plt.xlabel('Model')
plt.ylabel('Accuracy')

plt.subplot(1, 2, 2)
plt.bar(['Decision Tree'], [precision_dt], color='salmon')
plt.title('Precision Scores')
plt.xlabel('Model')
plt.ylabel('Precision')

# Show plot
plt.tight_layout()
plt.show()

# Print the results
print("Accuracy - ", accuracy_dt)
print("Confusion Matrix - \n", confusion_matrix_dt)
print("Precision - ", precision_dt)

# Plotting confusion matrix for Decision Tree
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_dt, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix - Decision Tree')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Pie chart for target variable distribution
plt.pie(df['target'].value_counts(), labels=['ham','spam'], autopct="%0.2f")
plt.show()


# In[ ]:





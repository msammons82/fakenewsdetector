#!/usr/bin/env python3
# made in jupyter labs 
import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score,confusion_matrix

# Read the data
df=pd.read_csv('C:/DataFlair/news.csv')

# Get shape and head
df.shape
df.head()

# Dataflair - get the labels
labels=df.label
labels.head()

# Dataflair - split the dataset
x_train,x_test,y_train,y_test = train_test_split(df['text'],labels,test_size=0.2, random_state=7)

#Dataflair - Initialize a TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english',max_df=0.7)

#Dataflair - Fit and transform train set, transform test set
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)

#Dataflair - Initialize a PassiveAggressiveClassifier
pac = PassiveAggressiveClassifier(max_iter= 50)
pac.fit(tfidf_test,y_train)

# Dataflair- Predict on the test set and calculate accuracy
y_pred= pac.predict(tfidf_test)
score = accuracy_score(y_test,y_pred)
print(f'Accuracy:  {round(score*100,2)}%')

# Dataflair- Build confusion matrix
confusion_matrix(y_test,y_pred, labels= ['FAKE','REAL'])


# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 03:54:13 2021

@author: Sumit
"""

#Importing the libraries
import numpy as np
import re
from nltk.corpus import stopwords
from sklearn.datasets import load_files
import nltk
import pickle
nltk.download('stopwords')


# Importing the dataset
reviews = load_files('data/')

# Storing data and target in different lists

X,y = reviews.data,reviews.target

# Loading files can take lot of time, so we will store the file in a pickle file so that it will be easy for us to use the data next time..

#Storing as Pickle Files
with open('X.pickle','wb') as file:
    pickle.dump(X,file)

with open('y.pickle','wb') as file:
    pickle.dump(y,file)


with open('X.pickle','rb') as file:
    X = pickle.load(file)
    
with open('y.pickle','rb') as file:
    y = pickle.load(file)
    
# Pre-processing and Creating Corpus
corpus = []
for i in range(len(X)):
    review = re.sub(r'\W',' ',str(X[i]))  #Replacig non-alphabets with space
    review = review.lower() # Converting to lowercase
    review = re.sub(r'\s+[a-z]\s+',' ',review) #Replacing single character with a space
    review = re.sub(r'^[a-z]\s+',' ',review) #Repalcing first single character with space
    review = re.sub(r'\s+',' ',review) #Replacing multiple space with single space
    review = re.sub(r'^\s+','',review) #Removing first space from the document
    corpus.append(review)
    

## Creating Bag of Words
'''
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=2000,
                             min_df=3,
                             max_df=0.6,
                             stop_words =stopwords.words('english')
                             )

# max_features ==> Denotes selection of 2000 most frequent words.
## min_df ==> Exclude words which appears in <= 3 documents
## max_df ==> Exclude words which appears in more than 60% of the document.

X = vectorizer.fit_transform(corpus).toarray()
'''
## Creating Tfidf Vectorizer from the corpus
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=2000,
                             min_df=3,
                             max_df=0.6,
                             stop_words =stopwords.words('english')
                             )
X = vectorizer.fit_transform(corpus).toarray()

## Transforming Already present BoW model into TF-IDF model
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer()
X = transformer.fit_transform(X).toarray()    



## Splitting into Train and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,stratify=y)

## Model Creation
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
y_pred = lr.predict(X_test)
print("Accuracy: ",accuracy_score(y_test,y_pred))
print("Precision: ",precision_score(y_test,y_pred))
print("Recall: ",recall_score(y_test,y_pred))
print("F-1 Score: ",f1_score(y_test,y_pred))

print("confusion matrix: ",confusion_matrix(y_test,y_pred))


## Saving th Classifier into a pickle file

with open("model.pickle","wb") as file:
    pickle.dump(lr,file)

## Saving the vectorizer soo that we can use it transform data for future prediction
with open("Vectorizer.pickle","wb") as file:
    pickle.dump(vectorizer,file)
    
# Using the saved model using any new data
with open("model.pickle","rb") as file:
    model = pickle.load(file)

with open("Vectorizer.pickle","rb") as file:
    vectorizer = pickle.load(file)
    
## Testing the saved model..
    
sample = ["You are bad person man, stay away from me."]
sample = vectorizer.transform(sample).toarray()
print(model.predict(sample))














    
    





















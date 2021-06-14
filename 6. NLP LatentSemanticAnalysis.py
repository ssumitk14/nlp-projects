# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 01:39:05 2021

@author: Sumit
"""

## Latent Semantic Analysis using Python

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

# SVD ==> Singular Value Decomposition
from sklearn.decomposition import TruncatedSVD

dataset = ["The amount of polution is increasing day by day",
           "The concert was just great",
           "I love to see Gordon Ramsay cook",
           "Google is introducing a new technology",
           "AI Robots are examples of great technology present today",
           "ALL of us were singing in the concert",
           "We have launch campaigns to stop pollution and global warming"
           ]

dataset = [lines.lower() for lines in dataset]

# TfidfVectorizer() can automatically valculate tfidf value of all the words.
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(dataset)

#Lets Visualize tfidf values for the 1st document
print(X[0])

# Decomposing the matrix "X" into 3 martixes
lsa = TruncatedSVD(n_components=4,n_iter=100)
# n_components ==> Total number of concepts to look for.
# Concepts is nothing but groups in which the each sentence is goig to be categorized to..For example: News, Trchnology, Music, etc

lsa.fit(X)

row1 = lsa.components_[0] 
#lsa.components_ will is the last matrix from the formula of SVD.
# It will contain words as rows and components as columns
# Each word will have their corresponding component value

# We willl get all the feature names(words) by using TfidfVectorizer
terms = vectorizer.get_feature_names()

# Now we will combine all the words with their corresponding component value.

concept_words = {}
for index,comp in enumerate(lsa.components_,start=1):
    componentTerms = zip(terms,comp)
    sortedTerms = sorted(componentTerms,key=lambda x:x[1],reverse=True)
    # Lets just printthe 10 most important word in each concept
    sortedTerms = sortedTerms[:10]
    print("Component",index,":")
    for word in sortedTerms:
        print(word)
    print("***********************************")
    
    concept_words["Concept"+str(index)] = sortedTerms

# Now we will add all the value of the words in each sentence to get a corresponding value of each sentence. This will give a score for each sentence belonging to respective component.

for key,values in concept_words.items():
    sentence_scores = []
    for sentence in dataset:
        words = nltk.word_tokenize(sentence)
        score = 0
        for word in words:
            for word_with_value in values:
                if word==word_with_value[0]:
                    score = score + word_with_value[1]
            
        sentence_scores.append(score)
    print("\n",key,":")
    for sentence_score in sentence_scores:
        print(sentence_score)




















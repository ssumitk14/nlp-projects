# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 03:28:29 2021

@author: Sumit
"""

import nltk
from nltk.corpus import wordnet

sentence = "I am not happy with the team's performance"

# not happpy ==> unhappy

words = nltk.word_tokenize(sentence)

new_sentence = []
temp_word = ""
for word in words:
    Antonyms = []
    if word == "not":
        temp_word = "not_"
    elif temp_word == "not_":
        for syn in wordnet.synsets(word):
            for s in syn.lemmas():
                for ant in s.antonyms():
                    Antonyms.append(ant.name())
        if(len(Antonyms)>=1):
            word = Antonyms[0]
        else:
            word = temp_word + word
        temp_word = ""
    
    if word != "not":
        new_sentence.append(word)
        
new_sentence = " ".join(new_sentence)
print("\nOriginal Sentence: ",sentence)
print("\nNew Sentence: ",new_sentence)
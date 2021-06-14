# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 03:18:50 2021

@author: Sumit
"""

import nltk
sentence = "I was not happy with the team's performance"

# We want to convert "not happy" as "not_happy"
words = nltk.word_tokenize(sentence)

new_sentence = []
temp_word = ''
for word in words:
    print(word)
    if word.lower()=="not":
        temp_word = "not_"
    elif temp_word == "not_":
        word = temp_word + word
        temp_word = ""
    
    if word!='not':
        new_sentence.append(word)
    
new_sentence = " ".join(new_sentence)

print(new_sentence)
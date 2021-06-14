# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 02:59:22 2021

@author: Sumit
"""

from nltk.corpus import wordnet

synonyms = []
antonyms = []

for syn in wordnet.synsets("good"):
    for s in syn.lemmas():
        print(s.name())
        synonyms.append(s.name())
        #print("------------Antonyms Starts here------------")
        for ant in s.antonyms():
            print(ant.name())
            antonyms.append(ant.name())
            
synonyms = list(set(synonyms))
antonyms = list(set(antonyms))



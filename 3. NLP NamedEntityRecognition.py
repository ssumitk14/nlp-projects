# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 19:22:19 2021

@author: Sumit
"""

import nltk

paragraph = "The Taj Mahal was built by Emperor Shah Jahan"

words = nltk.word_tokenize(paragraph)
tagged_words = nltk.pos_tag(words)


namedEntity = nltk.ne_chunk(tagged_words)
# This will create an object of type tree.Tree which we cannot visualize directly.
# To visualize the namedEntity We can use graph to visualize the named Entities


namedEntity.draw()
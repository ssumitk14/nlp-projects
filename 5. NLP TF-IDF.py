# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 16:30:23 2021

@author: Sumit
"""

import nltk

import re
import heapq # Used for selecting n largest elements from dictionary
from nltk.corpus import stopwords
import numpy as np
import math

paragraph = """It has finally happened, you guys. We have witnessed a historic moment. Well, we witnessed many historic moments, but Leonardo DiCaprio winning an Oscar for the first time just felt like a victory for all of us, you know? DiCaprio has been nominated for an Oscar so many times, only to go home empty-handed, that it has reached meme levels. So to see him finally stand on that stage with a golden statue in his hand was just so heartwarming. Unsurprisingly, when you look at the transcript of DiCaprio's Oscar acceptance speech (which he must surely have been planning for the last 10 years), you see how it's not just thanking people and then breezing off the stage. No, the 41-year-old actor took a moment to speak about a cause that's very close to his heart: environmentalism and climate change.I mean, don't get me wrong. DiCaprio did thank a lot of people, some of them collectively. (Like his friends. "You know who you are.") But working on The Revenant highlighted a cause that DiCaprio supports when the cameras stop rolling, and he wasn't going to let that go unmentioned in his speech. Not when he finally had the eyes of the world upon him for more than just making Sad Leo at the Oscars GIFs. Reading his speech should be enough to inspire fans to give environmentalism the same attention that we've given to DiCaprio's numerous losses over the years. However, even if it doesn't, the fact that he would draw attention away from himself and toward a good cause is just so DiCaprio that it really makes me feel like he deserved this award for more than one reason.You can watch the video of the speech here. Check out his full speech below:Thank you all so very much. Thank you to the Academy. Thank you to all of you in this room. I have to congratulate the other incredible nominees this year. The Revenant was the product of the tireless efforts of an unbelievable cast and crew. First off, to my brother in this endeavor, Mr. Tom Hardy. Tom, your talent on screen can only be surpassed by your friendship off screen … thank you for creating a transcendent cinematic experience. Thank you to everybody at Fox and New Regency … my entire team. I have to thank everyone from the very onset of my career … To my parents; none of this would be possible without you. And to my friends, I love you dearly; you know who you are.And lastly, I just want to say this: Making The Revenant was about man's relationship to the natural world. A world that we collectively felt in 2015 as the hottest year in recorded history. Our production needed to move to the southern tip of this planet just to be able to find snow. Climate change is real, it is happening right now. It is the most urgent threat facing our entire species, and we need to work collectively together and stop procrastinating. We need to support leaders around the world who do not speak for the big polluters, but who speak for all of humanity, for the indigenous people of the world, for the billions and billions of underprivileged people out there who would be most affected by this. For our children’s children, and for those people out there whose voices have been drowned out by the politics of greed. I thank you all for this amazing award tonight. Let us not take this planet for granted. I do not take tonight for granted. Thank you so very much."""

dataset = nltk.sent_tokenize(paragraph)

for i in range(len(dataset)):
    dataset[i] = dataset[i].lower()
    dataset[i] = re.sub(r"\W"," ",dataset[i])
    dataset[i] = re.sub(r"\s+"," ",dataset[i])
    
## Creating Bag of Words
word2count = {}
for data in dataset:
    words = nltk.word_tokenize(data)
    for word in words:
        if word not  in word2count.keys():
            word2count[word] = 1
        else:
            word2count[word] +=1


# Getting 100 most occuring words from the dictionary word2count
freq_words = heapq.nlargest(100,word2count,key=word2count.get)

## Creating word-IDF dictionary {'word':IDF Value} 
word_idf = {}
for word in freq_words:
    count = 0
    for data in dataset:
        if word in nltk.word_tokenize(data):
            count +=1
    word_idf[word] = math.log(len(dataset)/(count+1))  #We add 1 to the denominator to deal with zero division error.
    
    
    
# Creating TF dictionary {'word': TF Value}
word_tf = {}
for word in freq_words:
    tf = []
    for data in dataset:
        tf.append(nltk.word_tokenize(data).count(word)/len(nltk.word_tokenize(data)))
    word_tf[word] = tf
    
# Creating TF-IDF Matrix
TFIDF_matrix = []
for key,values in word_tf.items():
    tfidf = []
    for value in values:
        tfidf_value = value * word_idf[key]
        tfidf.append(tfidf_value)
    TFIDF_matrix.append(tfidf)
    

# Converting into 2d Array and taking transpose to make the words as columns with their respective tfidf values in rows.
X = np.asarray(TFIDF_matrix).T









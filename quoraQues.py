#Objective of the notebook is to introduce and build a naive bayes classifier from scratch for classifying text data.

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

train = pd.read_csv('./drive/My Drive/train.csv')
#sample_submission = pd.read_csv('../input/sample_submission.csv')
test = pd.read_csv('./drive/My Drive/test.csv')

print ('Shape of train ',train.shape)
print ('Shape of test ',test.shape)
#print ('Shape of sample_submission ',sample_submission.shape)

#In this part, we see what are sincere questions look like and how are insincere questions look like.

print ('Taking a look at Sincere Questions')
train.loc[train['target'] == 0].sample(5)['question_text']


print ('Taking a look at Insincere Questions')
train.loc[train['target'] == 1].sample(5)['question_text']

#Insincere questions are questions spreading hatred against a group of people or is not real. An insincere question has a value 1 while a sincere question has target value 0.

samp = train.sample(1)
sentence = samp.iloc[0]['question_text']
print (sentence)

#Text Preprocessing in Python

#Removing Numbers We use regex expressions to remove numbers.

import re
sentence = re.sub(r'\d+','',sentence)
print ('Sentence After removing numbers\n',sentence)

#Removing Punctuations in a string.

import string
sentence = sentence.translate(sentence.maketrans("","",string.punctuation))
print ('Sentence After Removing Punctuations\n',sentence)

#Removing Stop Words
#Stop words are the most common words in a language like “the”, “a”, “on”, “is”, “all”. These words do not carry important meaning and are usually removed from texts.
#Stop words can be removed by using Natural Language Toolkit (NLTK). NLTK is a set of libraries for symbolic ans statistical natural language processing. It was developed by University of Pennysylvania.

import nltk
nltk.download('punkt')
stop_words = set(nltk.corpus.stopwords.words('english'))
words_in_sentence = list(set(sentence.split(' ')) - stop_words)
print (words_in_sentence)

#Stemming of Words
#Stemming is the process for reducing derived words to their stem, base or root form—generally a written word form. Ex: owed -> owe muliply -> multipli

from nltk.stem import PorterStemmer
stemmer= PorterStemmer()
for i,word in enumerate(words_in_sentence):
    words_in_sentence[i] = stemmer.stem(word)
print (words_in_sentence)    

#Lemmatization of Words
#Lemmatisation is the process of grouping together the different inflected forms of a word so they can be analysed as a single item. Ex: dogs -> dog. I am not clear with difference between lemmatization and stemming. In most of the tutorials, I found them both and I could not understand the clear difference between the two.

from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
words = []
for i,word in enumerate(words_in_sentence):
    words_in_sentence[i] = lemmatizer.lemmatize(word)
print (words_in_sentence)


#Constructing a Naive Bayes Classifier from Scratch

from sklearn.model_selection import train_test_split
train, test = train_test_split(train, test_size=0.2)

"""
The next step are as follows:

Combine all the preprocessing techniques and create a dictionary of words and each word's count in training data.
Calculate probability for each word in a text and filter the words which has probability less than threshold probability. Words with probability less than threshold probability are insignificant.
Then for each word in the dictionary, I am creating a probability of that word being in insincere questions and its probability in sincere questions. I am finding the conditional probability to use in naive bayes classifier.
Prediction using condtional probabilities.
"""

#Step 1:
#Preprocessing training data. I create three dictionaries which hold word count of words occuring in sincere, insincere and overall after preprocessing each word.
word_count = {}
word_count_sincere = {}
word_count_insincere = {}
sincere  = 0
insincere = 0 

import re
import string
import nltk
stop_words = set(nltk.corpus.stopwords.words('english'))
from nltk.stem import PorterStemmer
stemmer= PorterStemmer()
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()

#for i in range(0,len(train.shape[0])):
#row_count = 1000
row_count = train.shape[0]
for row in range(0,row_count):
    insincere += train.iloc[row]['target']
    sincere += (1 - train.iloc[row]['target'])
    sentence = train.iloc[row]['question_text']
    sentence = re.sub(r'\d+','',sentence)
    sentence = sentence.translate(sentence.maketrans("","",string.punctuation))
    words_in_sentence = list(set(sentence.split(' ')) - stop_words)
    for index,word in enumerate(words_in_sentence):
        word = stemmer.stem(word)
        words_in_sentence[index] = lemmatizer.lemmatize(word)
    for word in words_in_sentence:
        if train.iloc[row]['target'] == 0:   #Sincere Words
            if word in word_count_sincere.keys():
                word_count_sincere[word]+=1
            else:
                word_count_sincere[word] = 1
        elif train.iloc[row]['target'] == 1:    #Insincere Words
            if word in word_count_insincere.keys():
                word_count_insincere[word]+=1
            else:
                word_count_insincere[word] = 1
        if word in word_count.keys():             #For all words. I use this to compute probability.
            word_count[word]+=1
        else:
            word_count[word]=1
            
print ('Done')

#Step 2
#Finding probability for each word in the dictionary.

word_probability = {}
total_words = 0
for i in word_count:
    total_words += word_count[i]
for i in word_count:
    word_probability[i] = word_count[i] / total_words

#Eliminating words which are insignificant. Insignificant words are words which have a probability of occurence less than 0.0001.
print ('Total words ',len(word_probability))
print ('Minimum probability ',min (word_probability.values()))
threshold_p = 0.0001
for i in list(word_probability):
    if word_probability[i] < threshold_p:
        del word_probability[i]
        if i in list(word_count_sincere):   #list(dict) return it;s key elements
            del word_count_sincere[i]
        if i in list(word_count_insincere):  
            del word_count_insincere[i]
print ('Total words ',len(word_probability))

#Step 3:
#To apply naive bayes algorithm, we have to find conditional probability. Finding the conditional probability.

total_sincere_words = sum(word_count_sincere.values())
cp_sincere = {}  #Conditional Probability
for i in list(word_count_sincere):
    cp_sincere[i] = word_count_sincere[i] / total_sincere_words

total_insincere_words = sum(word_count_insincere.values())
cp_insincere = {}  #Conditional Probability
for i in list(word_count_insincere):
    cp_insincere[i] = word_count_insincere[i] / total_insincere_words

#Step 4:
#Prediction
#for i in range(0,len(train.shape[0])):
#row_count = 1000
row_count = test.shape[0]

p_insincere = insincere / (sincere + insincere)
p_sincere = sincere / (sincere + insincere)
accuracy = 0

for row in range(0,row_count):
    sentence = test.iloc[row]['question_text']
    target = test.iloc[row]['target']
    sentence = re.sub(r'\d+','',sentence)
    sentence = sentence.translate(sentence.maketrans("","",string.punctuation))
    words_in_sentence = list(set(sentence.split(' ')) - stop_words)
    for index,word in enumerate(words_in_sentence):
        word = stemmer.stem(word)
        words_in_sentence[index] = lemmatizer.lemmatize(word)
    insincere_term = p_insincere
    sincere_term = p_sincere
    
    sincere_M = len(cp_sincere.keys())
    insincere_M = len(cp_insincere.keys())
    for word in words_in_sentence:
        if word not in cp_insincere.keys():
            insincere_M +=1
     #       print (word)
        if word not in cp_sincere.keys():
            sincere_M += 1
#        print (word)
         
    for word in words_in_sentence:
        if word in cp_insincere.keys():
            insincere_term *= (cp_insincere[word] + (1/insincere_M))
        else:
            insincere_term *= (1/insincere_M)
        if word in cp_sincere.keys():
            sincere_term *= (cp_sincere[word] + (1/sincere_M))
        else:
            sincere_term *= (1/sincere_M)
        
    if insincere_term/(insincere_term + sincere_term) > 0.5:
        response = 1
    else:
        response = 0
    if target == response:
        accuracy += 1
    
print ('Accuracy is ',accuracy/row_count*100)


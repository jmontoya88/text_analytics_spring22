#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 07:57:07 2023

@author: chestersmacbook
"""

#import all the packages that will be used in this lab
import matplotlib.pyplot as plt
import re
import os
import pandas as pd
import numpy as np

#processing libraries
import nltk
from nltk.tokenize import word_tokenize
from nltk import PorterStemmer
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords

#ML libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics

import gensim

#use pandas to read the csv file
data2 = pd.read_csv("clean_tweet_wiot_list.csv")
data = pd.read_csv("aspect_data.csv")

data['MESSAGE'] = data2.astype(str)
data['CATEGORY'] = data['y']

# remove non alphabets
remove_non_alphabets = lambda x: re.sub(r'[^a-zA-Z]',' ',x)

# tokenization
tokenize = lambda x: word_tokenize(x)

# stemming
ps = PorterStemmer()
stem = lambda w: [ ps.stem(x) for x in w ]

# lemmatization
lemmatizer = WordNetLemmatizer()
leammtizer = lambda x: [ lemmatizer.lemmatize(word) for word in x ]
'''
data['MESSAGE'] = data['tweets']
data['CATEGORY'] = (data['labels'] == 'good') | (data['labels'] == 'neutral')
'''


# apply all the methods above to the column Message
print('Processing : [=', end='')
data['MESSAGE'] = data['MESSAGE'].apply(remove_non_alphabets)
print('=', end='')
data['MESSAGE'] = data['MESSAGE'].apply(tokenize)
print('=', end='')
data['MESSAGE'] = data['MESSAGE'].apply(stem)
print('=', end='')
data['MESSAGE'] = data['MESSAGE'].apply(leammtizer)
print('=', end='')
data['MESSAGE'] = data['MESSAGE'].apply(lambda x: ' '.join(x))
print('] : Completed')






# split to 30 percent test data and 70 percent train data
# other splits of 60:40, 50:50 are also welcome

# labels are y, the dependent variable
# Messages are x, the dependent variable
train_corpus, test_corpus, train_labels, test_labels = train_test_split(data["MESSAGE"],
                                                                        data["CATEGORY"],
                                                                        test_size=0.3)

# build bag of words features' vectorizer and get features
bow_vectorizer = CountVectorizer(min_df=1, ngram_range=(1,1))
bow_train_features = bow_vectorizer.fit_transform(train_corpus)
bow_test_features = bow_vectorizer.transform(test_corpus)

# build tfidf features' vectorizer and get features
tfidf_vectorizer=TfidfVectorizer(min_df=1, 
                                 norm='l2',
                                 smooth_idf=True,
                                 use_idf=True,
                                 ngram_range=(1,1))
tfidf_train_features = tfidf_vectorizer.fit_transform(train_corpus)  
tfidf_test_features = tfidf_vectorizer.transform(test_corpus)

# tokenize documents for word2vec
tokenized_train = [nltk.word_tokenize(text)
                   for text in train_corpus]
tokenized_test = [nltk.word_tokenize(text)
                   for text in test_corpus] 

# build word2vec model                   
wv_model = gensim.models.Word2Vec(tokenized_train,
                               vector_size=200,                          #set the size or dimension for the word vectors 
                               window=60,                        #specify the length of the window of words taken as context
                               min_count=10)                   #ignores all words with total frequency lower than 10

def average_word_vectors(words, model, vocabulary, num_features):
    
    feature_vector = np.zeros((num_features,),dtype="float64")
    nwords = 0.
    
    for word in words:
        if word in vocabulary: 
            nwords = nwords + 1.
            feature_vector = np.add(feature_vector, model.wv[word])
    
    if nwords:
        feature_vector = np.divide(feature_vector, nwords)
        
    return feature_vector

def averaged_word_vectorizer(corpus, model, num_features):
    vocabulary = set(model.wv.index_to_key)
    features = [average_word_vectors(tokenized_sentence, model, vocabulary, num_features)
                    for tokenized_sentence in corpus]
    return np.array(features)

# averaged word vector features from word2vec
avg_wv_train_features = averaged_word_vectorizer(corpus=tokenized_train,
                                                 model=wv_model,
                                                 num_features=200)                   
avg_wv_test_features = averaged_word_vectorizer(corpus=tokenized_test,
                                                model=wv_model,
                                                num_features=200) 

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(avg_wv_train_features)
scaled_wv_train = scaler.transform(avg_wv_train_features)
scaler.fit(avg_wv_test_features)
scaled_wv_test = scaler.transform(avg_wv_test_features)

# define a function to evaluate our classification models based on four metrics
# This defined function is also useful in other cases. This is comparing test_y and pred_y. 
# Both contain 1s and 0s.
def get_metrics(true_labels, predicted_labels):
    
    print ('Accuracy:', np.round(                                                    
                        metrics.accuracy_score(true_labels, 
                                               predicted_labels),
                        2))
    print ('Precision:', np.round(
                        metrics.precision_score(true_labels, 
                                               predicted_labels),
                        2))
    print ('Recall:', np.round(
                        metrics.recall_score(true_labels, 
                                               predicted_labels),
                        2))
    print ('F1 Score:', np.round(
                        metrics.f1_score(true_labels, 
                                               predicted_labels),
                        2))
    
# define a function that trains the model, performs predictions and evaluates the predictions
def train_predict_evaluate_model(classifier, 
                                 train_features, train_labels, 
                                 test_features, test_labels):
    # build model    
    classifier.fit(train_features, train_labels)
    # predict using model
    predictions = classifier.predict(test_features) 
    # evaluate model prediction performance   
    '''get_metrics(true_labels=test_labels, 
                predicted_labels=predictions)'''
    print(metrics.classification_report(test_labels,predictions))
    get_metrics(test_labels,predictions)

    return predictions, metrics.accuracy_score(test_labels,predictions),   metrics.confusion_matrix(test_labels,predictions)

from sklearn.naive_bayes import MultinomialNB # import naive bayes
from sklearn.tree import DecisionTreeClassifier # import Decision Tree
from sklearn.ensemble import RandomForestClassifier # import random forest


mnb = MultinomialNB()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier(criterion="entropy")
model_dic ={'mnb':mnb, 'dt':dt, 'rf':rf}

feature_dic = {'bow':[bow_train_features,bow_test_features], 
               'tfidf':[tfidf_train_features,tfidf_test_features],
               'scaled_wv':[scaled_wv_train,scaled_wv_test]}


dic_predictions={} 
dic_accuracy={}
dic_confusion_matrix={}

for model_name in model_dic:
    dic_predictions[model_name]={} 
    dic_accuracy[model_name]={}
    dic_confusion_matrix[model_name]={}

for model_name in model_dic:
    for feature_name in feature_dic:
        model = model_dic[model_name]
        train_feature = feature_dic[feature_name][0]
        test_feature = feature_dic[feature_name][1]
        
        print("Now Running feature: {}".format(feature_name))
        print("Now Running modeling: {}".format(model_name))
        #print(f"Now Running {feature_name}")
        print(feature_name, model_name)
        
        predictions, accuracy, confusion_matrix = train_predict_evaluate_model(classifier=model,
                                           train_features=train_feature,
                                           train_labels=train_labels,
                                           test_features=test_feature,
                                           test_labels=test_labels)
        
        dic_predictions[model_name][feature_name]=predictions
        dic_accuracy[model_name][feature_name]=accuracy
        dic_confusion_matrix[model_name][feature_name]=confusion_matrix
        #print("Now Ending {}".format(feature_name))
        print("====================================================================================================")
        print()

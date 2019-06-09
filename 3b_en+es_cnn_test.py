#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 16:46:13 2019

@author: macbook
"""

import os
import re
import sys
import pickle
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences

    
from keras.models import load_model
from nltk.tokenize import RegexpTokenizer

from pymagnitude import *

def tokenize(sentence):
    wordsList=[]
    tokenizer = RegexpTokenizer(r'[^\d\W]+')
    tokens = tokenizer.tokenize(sentence)
    wordsList+=tokens
    return wordsList

    
def cleaned_text(docs):
    docs = docs.lower()
    docs = re.sub(r'[^A-Za-z ]+', '', docs)
    return docs

def get_idx_n_pad(tokens_column, dictionary, maxlen):
    texts_idx_list = []
    for i in range(tokens_column.shape[0]):
        idx_list = []
        for token in tokens_column.iloc[i]:
            word_idx = 0
            if token in dictionary:
                word_idx = dictionary[token]
            idx_list.append(word_idx)
            if len(idx_list) > maxlen:
                break
        texts_idx_list.append(idx_list)
    texts_idx_list = pad_sequences(texts_idx_list, maxlen)
    return texts_idx_list

if __name__ == '__main__':
    
    with open("./test/english_test.text") as f:
        texts = [l.strip() for l in f]
    with open("./test/english_test.labels") as f:
        labels = [l.strip() for l in f]

    Labelvec = []    
    for lab in labels:
        tmp = np.zeros(len(set(labels)))
        tmp[int(lab)]=1.0
        Labelvec.append(tmp)
    
    df = pd.DataFrame(list(zip(Labelvec, texts)), columns =['Label', 'Text'])
    df['CLEANED_TEXT'] = df.Text.apply(cleaned_text)
    df['TOKENS'] = df.CLEANED_TEXT.apply(tokenize)
    
    input_dictionary = pickle.load(open('./static/en+es_dictionary.pkl', 'rb'))
            
    MAX_SEQUENCE_LENGTH = 50

    X_test_padded = get_idx_n_pad(df['TOKENS'], input_dictionary, MAX_SEQUENCE_LENGTH)
    y_test = np.array(df['Label'].tolist())

    es_cnn_model = load_model('./static/en+es_cnn_model.h5')
    p_v = es_cnn_model.predict(X_test_padded)
    pred = [np.argmax(p) for p in p_v]
       
    #create a submission
    with open('./submission/en+es_cnn_test_submission', 'w+') as submission_file:
        for p in pred:
            submission_file.write(str(p)+'\n')
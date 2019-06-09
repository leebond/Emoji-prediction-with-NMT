#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 00:32:04 2019

@author: macbook
"""


import os
import re
import sys
import math
import pickle
import collections
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Flatten, Dropout, Embedding, LSTM, TimeDistributed, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, Activation
from keras.models import load_model
from keras.callbacks import EarlyStopping
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

def build_dataset(words, n_words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

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
    
    with open("./train/spanish_train.text") as f:
        texts = [l.strip() for l in f]
    with open("./train/spanish_train.labels") as f:
        labels = [l.strip() for l in f]
    with open("./mapping/spanish_mapping.txt") as f:
        mappings = [l.strip() for l in f]
    
    Labelvec = []
    for lab in labels:
        tmp = np.zeros(len(set(labels)))
        tmp[int(lab)]=1.0
        Labelvec.append(tmp)
    
    df = pd.DataFrame(list(zip(Labelvec, texts)), columns =['Label', 'Text'])
    df['CLEANED_TEXT'] = df.Text.apply(cleaned_text)
    df['TOKENS'] = df.CLEANED_TEXT.apply(tokenize)
    
    texts = df['TOKENS'].tolist()
    vocabulary_size = 10000
    vocab = [t for t in texts for t in t]
    dim = 300
    n_words = vocabulary_size
    
    data_index = 0
    input_data, input_count, input_dictionary, input_reverse_dictionary = build_dataset(vocab, vocabulary_size)
    
    FASTTEXT_FNAME = './static/wiki.es.magnitude'
    embeddings_index = Magnitude(FASTTEXT_FNAME)
            
    MAX_SEQUENCE_LENGTH = 50
    
    embedding_matrix = np.zeros((len(input_dictionary) + 1, dim))
    for word, i in input_dictionary.items():
        embedding_vector = embeddings_index.query(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    
    embedding_layer = Embedding(embedding_matrix.shape[0], # or len(word_index) + 1
                                embedding_matrix.shape[1], # or EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)

    X_train_padded = get_idx_n_pad(df['TOKENS'], input_dictionary, MAX_SEQUENCE_LENGTH)
    y_train = np.array(df['Label'].tolist())
    
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(y_train.shape[1], activation='softmax')(x)
    cnn_model = Model(sequence_input, preds)
    cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    print(cnn_model.summary())
    
    #define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, verbose=1)
    callbacks_list = [early_stopping]
    
    cnn_model.fit(X_train_padded, y_train, batch_size = 128, epochs = 16)
    
    cnn_model.save('./static/es_cnn_model.h5')
    cnn_model = load_model('./static/es_cnn_model.h5')
    p_v = cnn_model.predict(X_train_padded)
    pred = [np.argmax(p) for p in p_v]
       
    #create a submission
    with open('./submission/es_cnn_submission', 'w+') as submission_file:
        for p in pred:
            submission_file.write(str(p)+'\n')

    with open('./static/es_dictionary' + '.pkl', 'wb') as f:
        pickle.dump(input_dictionary, f, pickle.HIGHEST_PROTOCOL)
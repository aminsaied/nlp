# python 3
"""Contains the Tic and Toc classes used to detect clusters in text-data.

This is a rather strange idea I wanted to try out. Here are the two steps:

1) Tic:
    - Emb corpus of text into a vector-space using a w2v embedding matrix
    - Use k-means to detect clusters
    - Label the documents with their cluster index
    - Pass this labelled data to the toc-step

2) Toc:
    - Use the embedding matrix from the tic-step to build an RNN
    - Train a RNN to predict the cluster labels
    - Update the embedding matrix during training
    - Pass this updated embedding matrix to the tic-step

Please be aware that this is not an established technique - just a fun
experiment!
"""

from __future__ import print_function

import numpy as np
# import pandas as pd

# imports for TIC-step
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
from sklearn import metrics

# imports for TOC-step
from keras.models import Model
from keras.layers import Dense, Input, Bidirectional, GRU
from keras.layers.embeddings import Embedding

# surpress some tensorflow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from utils import *

class Tic(object):
    """The tic step in the tic-toc algorithm.

    Tic-step:
        - Emb corpus of text into a vector-space using a w2v embedding matrix
        - Use k-means to detect clusters
        - Label the documents with their cluster index
        - Pass this labelled data to the toc-step
    """

    def __init__(self, arxiv, emb_matrix, X_indices, Y_labels):
        self.arxiv = arxiv
        self.emb_matrix = emb_matrix
        self.labels = Y_labels
        self.X_indices = X_indices

        self.tic_params = {
            'max_len'   : X_indices.shape[1],
            'n_clusters': len(list(set(Y_labels)))
        }

        self.indices_to_embedding = self._build_indices_to_embedding(emb_matrix)
        self._normalizer = Normalizer(copy=False)

        self.v_measure_history = []

    def set_embedding_matrix(self, emb_matrix):
        self.indices_to_embedding.set_weights([emb_matrix])

    def update_clusters(self):
        # embed corpus
        X_w2v = self._embed_corpus_in_w2v_space()

        # fit k-means
        n_clusters = self.tic_params['n_clusters']
        km = KMeans(n_clusters)
        km.fit(X_w2v)

        # update labels
        self.arxiv['Code'] = km.labels_

        # evaluate againt Y_labels
        v_measure = metrics.v_measure_score(self.labels, km.labels_)
        self.v_measure_history.append(v_measure)

        return self.arxiv

    def _embed_corpus_in_w2v_space(self):
        X_w2v = self.indices_to_embedding.predict(self.X_indices)
        X_w2v = np.mean(X_w2v, axis=1)
        return self._normalizer.transform(X_w2v)

    def _build_indices_to_embedding(self, emb_matrix):
        vocab_len, emb_dim = emb_matrix.shape

        embedding_layer = Embedding(input_dim = vocab_len,
                                    output_dim = emb_dim,
                                    trainable = False)

        embedding_layer.build((None,))
        embedding_layer.set_weights([emb_matrix])

        max_len = self.tic_params['max_len']
        sentence_indices = Input(shape=(max_len,), dtype='int32')
        E = embedding_layer(sentence_indices)

        return Model(inputs=sentence_indices, outputs=E)

class Toc(object):
    """The toc step in the tic-toc algorithm.

    Toc-step:
        - Use the embedding matrix from the tic-step to build an RNN
        - Train a RNN to predict the cluster labels
        - Update the embedding matrix during training
        - Pass this updated embedding matrix to the tic-step
    """

    def __init__(self, X_indices, emb_matrix, n_classes, n_test, params=None):

        self.X_indices  = X_indices
        self.emb_matrix = emb_matrix
        self.n_classes  = n_classes
        self.n_test     = n_test
        self.accuracy_history = []

        self.toc_params = {
            'max_len'   : 100,
            'n_epochs'  : 2,
            'batch_size': 64,
            'n_hidden'  : 128,
            'p_drop'    : 0.5
        }

        if params:
            self.set_params(params)

        self._build_model()

    def set_params(self, params):
        for key in params:
            if key in self.toc_params:
                self.toc_params[key] = params[key]

    def set_clusters(self, arxiv):
        # build output matrix
        Y = self._build_output(arxiv)

        # split training/test data
        self.train, self.test = self._split_data(self.X_indices, Y)

    def update_embedding_matrix(self):
        # get params
        n_epochs   = self.toc_params['n_epochs']
        batch_size = self.toc_params['batch_size']

        # train RNN
        X_train, Y_train = self.train
        self.model.fit(X_train, Y_train, epochs = n_epochs,
                                     batch_size = batch_size, shuffle=True)

        # update model history
        X_test, Y_test   = self.test
        loss, acc = self.model.evaluate(X_test, Y_test)
        self.accuracy_history.append(acc)

        # get updated embedding matrix
        emb_matrix_step = self.model.get_weights()[0]

        return emb_matrix_step

    def _build_model(self):

        # useful varaibles
        vocab_len, emb_dim = self.emb_matrix.shape
        max_len  = self.toc_params['max_len']
        n_hidden = self.toc_params['n_hidden']
        p_drop   = self.toc_params['p_drop']
        K = self.n_classes

        # Build embedding layer
        embedding_layer = Embedding(input_dim = vocab_len,
                                    output_dim = emb_dim,
                                    trainable = True) # we want to update the embedding matrix
        embedding_layer.build((None,))
        embedding_layer.set_weights([self.emb_matrix])

        # build model
        sentence_indices = Input(shape=(max_len,), dtype='int32')

        E = embedding_layer(sentence_indices)
        X = Bidirectional(GRU(n_hidden, dropout=p_drop, return_sequences=True))(E)
        X = Bidirectional(GRU(n_hidden, dropout=p_drop))(X)
        X = Dense(units = K, activation='softmax')(X)

        self.model = Model(inputs=sentence_indices, outputs=X)

        # compile model
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

    def _build_output(self, arxiv):
        Y_indices = np.asarray( arxiv['Code'] )
        m, K = len(Y_indices), self.n_classes

        Y = np.zeros((m, K))
        for i in range(m):
            j = Y_indices[i]
            Y[i][j] = 1

        return Y

    def _split_data(self, X_indices, Y):
        n_test = self.n_test
        X_train, X_test = X_indices[:-n_test], X_indices[-n_test:]
        Y_train, Y_test = Y[:-n_test], Y[-n_test:]
        return (X_train, Y_train), (X_test, Y_test)

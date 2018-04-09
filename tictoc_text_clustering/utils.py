#!/usr/bin/env python3
"""Helper functions for `tictoc_experiment` script.
"""
import numpy as np
import pandas as pd

# useful variables
DIR = 'data/'
EXT = '.pkl'
MAX_LEN = 100

def load_data(max_len=MAX_LEN):
    """Return data to run tic-toc experiment."""
    # read data from disk
    arxiv = pd.read_pickle( DIR + 'arxiv' + EXT)
    word_to_vec = pd.read_pickle(DIR + 'w2v_dict_50' + EXT)
    msc_to_index = pd.read_pickle(DIR + 'msc_to_index' + EXT)

    # construct embedding matrix
    word_to_index = word_indices(word_to_vec)
    emb_matrix = embedding_matrix(word_to_vec, word_to_index)

    # construct input/output data
    sentences = np.asarray( arxiv['Abstract'] )
    X_indices = sentences_to_indices(sentences, word_to_index, max_len)
    Y_labels = np.asarray([msc_to_index[x] for x in arxiv['Code'].tolist()])

    return arxiv, emb_matrix, X_indices, Y_labels

def embedding_matrix(word_to_vec, word_to_index):
    """Constructs embedding matrix from word_to_vec model."""
    vocab_len = len(word_to_index) + 1              # keras requirement
    emb_dim = word_to_vec['group'].shape[0]         # dim of word vectors
    emb_matrix = np.zeros((vocab_len, emb_dim))     # initialise ourput array

    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec[word]

    return emb_matrix

def sentences_to_indices(sentences, word_to_index, max_len):
    """Convert sentences to array of indices."""
    m = sentences.shape[0]              # num training examples
    X = np.zeros((m, max_len))          # initialise output array

    for i in range(m):
        sentence = sentences[i]
        indices = [word_to_index[w] for w in sentence[:max_len]]
        X[i] = indices + [0] * (max_len-len(indices))

    return X

def word_indices(word_to_vec):
    """Assign unique index to each word in vocab."""
    vocab = list(word_to_vec.keys())
    vocab.sort()
    word_to_index = {word: ix for (ix,word) in enumerate(vocab)}
    return word_to_index

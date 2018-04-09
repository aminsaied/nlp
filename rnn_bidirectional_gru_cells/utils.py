#!/usr/bin/env python3
"""Helper functions for `rnn_bidirectional_gru` script.
"""
import numpy as np
import pandas as pd

# useful variables
DIR = 'data/'
EXT = '.pkl'

def arxiv_dataset(max_len=100, n_test=2000):
    """Loads data ready to train our model.

    Args:
        max_len: int, maximum number of words considered in each input
        n_test: int, the size of the test data

    Return:
        arxiv: DataFrame, containing the abstracts and their codes
        emb_matrix: array, mapping one-hot to w2v-embeddings
        X_train, Y_train: tuple of training data
        X_test, Y_test: tuple of test data
    """
    # load data from disk
    arxiv = pd.read_pickle( DIR + 'arxiv' + EXT)
    word_to_vec = pd.read_pickle(DIR + 'w2v_dict_50' + EXT)
    msc_to_index = pd.read_pickle(DIR + 'msc_to_index' + EXT)

    # assign indices to words
    vocab = list(word_to_vec.keys())
    vocab.sort()
    word_to_index = {word: ix for (ix,word) in enumerate(vocab)}

    # build the embedding matrix
    emb_matrix = embedding_matrix(word_to_vec, word_to_index)

    # create input data
    sentences = np.asarray( arxiv['Abstract'] )
    X_indices = sentences_to_indices(sentences, word_to_index, max_len)

    # create output data
    primary_classes = np.asarray( arxiv['Code'] )
    Y_indices = primary_to_indices(primary_classes, msc_to_index)

    # split training/test data
    X_train, X_test = X_indices[:-n_test], X_indices[-n_test:]
    Y_train, Y_test = Y_indices[:-n_test], Y_indices[-n_test:]

    return arxiv, emb_matrix, (X_train, Y_train), (X_test, Y_test)

def embedding_matrix(word_to_vec, word_to_index):
    """Return an embedding matrix from a word_to_vec dict.

    Args:
        word_to_vec: dict, mapping words to their embeddings

    Return:
        An array with rows corresponding to word embeddings.
    """
    vocab_len = len(word_to_index) + 1              # Keras requirement
    emb_dim = word_to_vec['group'].shape[0]         # dim of word vectors

    # build embedding matrix
    emb_matrix = np.zeros((vocab_len, emb_dim))
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec[word]

    return emb_matrix

def sentences_to_indices(sentences, word_to_index, max_len):
    """Convert sentence into array of indices.

    Args:
        sentences: array, of shape (m, 1) made of `m` sentences
        word_to_index: dict, mapping words to their indices
        max_len: int, the maximum length of a sentence

    Return:
        An array of shape (m, max_len) consisting of word indices.
    """
    m = sentences.shape[0]                  # num training examples
    X = np.zeros((m, max_len))              # initialise output array

    for i in range(m):
        sentence = sentences[i]
        indices = [word_to_index[w] for w in sentence[:max_len]]
        X[i] = indices + [0] * (max_len-len(indices))

    return X

def primary_to_indices(primary_classes, msc_index):
    """Convert msc-code to its index.
    """
    m = primary_classes.shape[0] # number of training examples
    K = len(msc_index) # number of classes

    Y = np.zeros((m, K)) # initialise matrix of indices

    for i in range(m):

        primary_class = primary_classes[i]
        j = msc_index[primary_class]
        Y[i][j] = 1

    return Y

#!/usr/bin/env python3
"""Helper functions for `comparison_of_document_embeddings` script.
"""
import numpy as np
import pandas as pd

DIR = 'data/'
EXT = '.pkl'

def load_dataset():
    arxiv = pd.read_pickle(DIR + 'arxiv' + EXT)
    codes = pd.read_pickle(DIR + 'msc_to_index' + EXT)
    w2v = pd.read_pickle(DIR + 'w2v_dict_50' + EXT)
    return arxiv, codes, w2v

def build_vocab_from_corpus(corpus):
    """Return a vocab from a list of abstracts.

    Args:
        corpus: a list of abstracts, where an abstract is a list of words

    Return:
        dict: mapping words to an index
    """
    assert type(corpus) == list
    assert len(corpus) > 0
    assert type(corpus[0]) == list

    vocab = [word for ab in corpus for word in ab]
    vocab = list(set(vocab))
    vocab.sort()
    vocab = {word:idx for (idx, word) in enumerate(vocab)}

    return vocab

def k_means_data_as_one_hot_from_df(df, vocab, col_name = 'Abstract'):
    """Return array of training data for K-means consisting of one-hots.

    Args:
        df: dataframe, containing column `col_name`
        vocab: dict, mapping words to indices
        col_name: str, name of column in `df` containing abstracts

    Return:
        array: of shape (len(df), len(vocab)) with one-hot representatives.
    """
    assert (col_name in df.columns)

    # for K-means in scikit-learn the data should be
    # of the shape (length_of_data, dim_of_vector_space)
    n_papers = len(df)
    n_words = len(vocab)
    X = np.zeros((n_papers, n_words))

    # populate X with one-hot representatives
    for idx, paper in df.iterrows():
        ab_oh = one_hot_of_abstract(paper[col_name], vocab)
        X[idx] = ab_oh

    return X

def k_means_data_as_w2v_from_corpus(corpus, w2v):
    """Return array of training data for K-means consisting of w2v embeddings.

    Args:
        corpus: list, of abstracts
        w2v: dict, mapping words to w2v embeddings

    Return:
        array: of shape (len(corpus), dim_w2v).
    """
    # for K-means in scikit-learn the data should be
    # of the shape (length_of_data, dim_of_vector_space)
    n_papers = len(corpus)
    dim_w2v = compute_dim_w2v(w2v)
    X_w2v = np.zeros((n_papers, dim_w2v))

    # populate X with one-hot representatives
    for idx, ab in enumerate(corpus):
        ab_w2v = w2v_of_abstract(ab, w2v)
        X_w2v[idx] = ab_w2v

    return X_w2v

def one_hot_of_abstract(ab, vocab):
    """Return one-hot representation of an abstract.

    Args:
        ab: list, of words
        vocab: dict, mapping words to indices

    Return:
        array: sum of one-hot vectors of each word in `ab`
    """
    n_words = len(vocab)
    oh = np.zeros(n_words)
    for word in ab:
        idx = vocab[word]
        oh[idx] += 1
    return oh

def compute_dim_w2v(w2v):
    """Return the dimension of word embeddings in w2v dict."""
    a_word =  list(w2v.keys())[0]
    return w2v[a_word].shape[0]

def w2v_of_abstract(ab, w2v):
    """Return word-to-vec representation of an abstract."""
    dim_w2v = compute_dim_w2v(w2v)
    ab_w2v = np.zeros(dim_w2v)
    for word in ab:
        if word in w2v:
            emb_word = w2v[word]
            ab_w2v += emb_word
    return ab_w2v

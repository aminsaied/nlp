# python 3

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn import metrics

DIR = 'data/'
EXT = '.pkl'

def load_data(max_len = 100):
    arxiv = pd.read_pickle( DIR + 'arxiv' + EXT)
    word_to_vec = pd.read_pickle(DIR + 'w2v_dict_50' + EXT)
    msc_to_index = pd.read_pickle(DIR + 'msc_to_index' + EXT)

    # assign indices to words
    vocab = list(word_to_vec.keys())
    vocab.sort()
    word_to_index = {word: ix for (ix,word) in enumerate(vocab)}

    # get output labels
    Y_labels = np.asarray([msc_to_index[x] for x in arxiv['Code'].tolist()])

    # build vocab
    vocab = list(word_to_vec.keys())
    vocab.sort()
    word_to_index = {word: ix for (ix,word) in enumerate(vocab)}

    # build the embedding matrix
    emb_matrix = _embedding_matrix(word_to_vec, word_to_index)

    # create input data
    sentences = np.asarray( arxiv['Abstract'] )
    X_indices = sentences_to_indices(sentences, word_to_index, max_len)

    return arxiv, emb_matrix, X_indices, Y_labels

# def training_data(df, X_indices):
#     Y_indices = _build_output_indices(df, col_name='Code')
#     return _train_test_split(X_indices, Y_indices, n_test=2000)

def _build_output_indices(df, col_name='Code'):
    # create output data
    primary_classes = np.asarray( arxiv['Code'] )
    return primary_to_indices(primary_classes, msc_to_index)

def _train_test_split(X_indices, Y_indices, n_test=2000):
    X_train, X_test = X_indices[:-n_test], X_indices[-n_test:]
    Y_train, Y_test = Y_indices[:-n_test], Y_indices[-n_test:]
    return (X_train, Y_train), (X_test, Y_test)

# def k_means_data_from_embedding_matrix(X_indices, emb_matrix):
#     pass

# def k_means_data_from_df(df, w2v, col_name = 'Abstract'):
#     """Return array of training data for K-means consisting of w2v embeddings.
#
#     Args:
#         df: dataframe, containing column `col_name`
#         w2v: dict, mapping words to w2v embeddings
#         col_name: str, name of column in `df` containing abstracts
#
#     Return:
#         array: of shape (len(df), dim(w2v)).
#     """
#
#     assert (col_name in df.columns)
#
#     # for K-means in scikit-learn the data should be
#     # of the shape (length_of_data, dim_of_vector_space)
#     n_papers = len(df)
#     dim_w2v = _compute_dim_w2v(w2v)
#     X_w2v = np.zeros((n_papers, dim_w2v))
#
#     # populate X with one-hot representatives
#     for idx, paper in df.iterrows():
#         ab_w2v = _w2v_of_abstract(paper[col_name], w2v)
#         X_w2v[idx] = ab_w2v
#
#     return X_w2v



# def build_vocab_from_corpus(corpus):
#     """Return a vocab from a list of abstracts.
#
#     Args:
#         corpus: a list of abstracts, where an abstract is a list of words
#
#     Return:
#         dict: mapping words to an index
#     """
#
#     assert type(corpus) == list
#     assert len(corpus) > 0
#     assert type(corpus[0]) == list
#
#     vocab = [word for ab in corpus for word in ab]
#     vocab = list(set(vocab))
#     vocab.sort()
#     vocab = {word:idx for (idx, word) in enumerate(vocab)}
#
#     return vocab

def _compute_dim_w2v(w2v):
    """Return the dimension of word embeddings in w2v dict."""
    a_word =  list(w2v.keys())[0]
    return w2v[a_word].shape[0]

def _w2v_of_abstract(ab, w2v):
    """Return w2v representation of an abstract.

    Args:
        ab: list, of words
        w2v: dict, mapping words to w2v embeddings

    Return:
        array: sum of w2v embeddings of each word in `ab`
    """
    dim_w2v = _compute_dim_w2v(w2v)
    ab_w2v = np.zeros(dim_w2v)
    for word in ab:
        if word in w2v:
            emb_word = w2v[word]
            ab_w2v += emb_word
    return ab_w2v



def _embedding_matrix(word_to_vec, word_to_index):
    """Return an embedding matrix from a word_to_vec dict.
    Arguments:
    word_to_vec -- a dictionary containing words mapped to their embeddings
    """
    # adding 1 to fit Keras embedding (requirement)
    vocab_len = len(word_to_index) + 1

    # define dimensionality of word vectors
    emb_dim = word_to_vec['group'].shape[0]

    # Initialize the embedding matrix
    emb_matrix = np.zeros((vocab_len, emb_dim))

    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec[word]

    return emb_matrix

def sentences_to_indices(sentences, word_to_index, max_len):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to `Embedding()`.

    Arguments:
    sentences -- array of sentences (strings), of shape (m, 1)
    word_to_index -- a dictionary containing words mapped to their index
    max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this.

    Returns:
    X -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """

    m = sentences.shape[0] # number of training examples
    X = np.zeros((m, max_len))

    for i in range(m):

        sentence = sentences[i]

        indices = [word_to_index[w] for w in sentence[:max_len]]
        X[i] = indices + [0] * (max_len-len(indices))

    return X

def primary_to_indices(primary_classes, msc_index):

    m = primary_classes.shape[0] # number of training examples
    K = len(msc_index) # number of classes

    Y = np.zeros((m, K)) # initialise matrix of indices

    for i in range(m):

        j = primary_classes[i]
        Y[i][j] = 1

    return Y

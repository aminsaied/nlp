# python 3

import numpy as np
import pandas as pd
# from collection import Counter

from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn import metrics

# useful variables
DATA_DIR = 'data/'
EXT = '.pkl'
DF_NAME = 'arxiv'


def load_dataset():
    df = pd.read_pickle(DATA_DIR + DF_NAME + EXT)
    return df

class DocumentFeatures(object):

    def __init__(self, corpus, max_features = 10000, n_lsa = 20, verbose = False):
        """
        Args:
            corpus: list, of strings corresponding to documents
            max_features: int (default 10000), how many words to keep in vocab
            n_lsa: int (default 20), how many lsa features to extract
            verbose: bool, print info to console
        """
        self.corpus = corpus
        self.max_features = max_features
        self.n_lsa = n_lsa
        self.verbose = verbose

        self.X_tfidf = None
        self.X_lsa = None

        self._build_tfidf()
        self._build_lsa()

    def _build_tfidf(self):
        if self.verbose:
            print('Building TFIDF features...')


        vectorizer = TfidfVectorizer(max_df=0.5, min_df=2,
                                     max_features=self.max_features,
                                     stop_words='english',
                                     use_idf=True)

        self.X_tfidf = vectorizer.fit_transform(self.corpus)

    def _build_lsa(self):
        if self.verbose:
            print('Building LSA features...')

        svd = TruncatedSVD(self.n_lsa)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)

        assert self.X_tfidf is not None
        self.X_lsa = lsa.fit_transform(self.X_tfidf)

# def find_min_count(df, col_name):
#     code_counter = Counter(df[col_name].tolist())
#     return code_counter.most_common()[-1][1]


def build_corpus_from_df(df, col_name):
    to_string = lambda ab: ' '.join(ab)
    return df[col_name].apply(to_string).tolist()

def build_vocab_from_corpus(ab_list):
    """Return a vocab from a list of abstracts.

    Args:
        ab_list: a list of abstracts, where an abstract is a list of words

    Return:
        dict: mapping words to an index
    """

    assert type(ab_list) == list
    assert len(ab_list) > 0
    assert type(ab_list[0]) == list

    vocab = [word for ab in ab_list for word in ab]
    vocab = list(set(vocab))
    vocab.sort()
    vocab = {word:idx for (idx, word) in enumerate(vocab)}

    return vocab

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

def _compute_dim_w2v(w2v):
    """Return the dimension of word embeddings in w2v dict."""
    a_word =  list(w2v.keys())[0]
    return w2v[a_word].shape[0]

def w2v_of_abstract(ab, w2v):
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

def k_means_data_as_one_hot_from_abstracts(ab_list, vocab):
    """Return array of training data for K-means consisting of one-hots.

    Args:
        ab_list: list, of abstracts
        vocab: dict, mapping words to indices

    Return:
        array: of shape (len(ab_list), len(vocab)) with one-hot representatives.
    """

    # for K-means in scikit-learn the data should be
    # of the shape (length_of_data, dim_of_vector_space)
    n_papers = len(ab_list)
    n_words = len(vocab)
    X = np.zeros((n_papers, n_words))

    # populate X with one-hot representatives
    for idx, ab in enumerate(ab_list):
        ab_oh = one_hot_of_abstract(ab, vocab)
        X[idx] = ab_oh

    return X

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

def k_means_data_as_w2v_from_abstracts(ab_list, w2v):
    """Return array of training data for K-means consisting of w2v embeddings.

    Args:
        ab_list: list, of abstracts
        w2v: dict, mapping words to w2v embeddings

    Return:
        array: of shape (len(ab_list), dim_w2v).
    """

    # for K-means in scikit-learn the data should be
    # of the shape (length_of_data, dim_of_vector_space)
    n_papers = len(ab_list)
    dim_w2v = _compute_dim_w2v(w2v)
    X_w2v = np.zeros((n_papers, dim_w2v))

    # populate X with one-hot representatives
    for idx, ab in enumerate(ab_list):
        ab_w2v = w2v_of_abstract(ab, w2v)
        X_w2v[idx] = ab_w2v

    return X_w2v

def k_means_data_as_w2v_from_df(arxiv, w2v, col_name = 'Abstract'):
    """Return array of training data for K-means consisting of w2v embeddings.

    Args:
        df: dataframe, containing column `col_name`
        w2v: dict, mapping words to w2v embeddings
        col_name: str, name of column in `df` containing abstracts

    Return:
        array: of shape (len(df), dim(w2v)).
    """

    assert (col_name in df.columns)

    # for K-means in scikit-learn the data should be
    # of the shape (length_of_data, dim_of_vector_space)
    n_papers = len(df)
    dim_w2v = _compute_dim_w2v(w2v)
    X_w2v = np.zeros((n_papers, dim_w2v))

    # populate X with one-hot representatives
    for idx, paper in df.iterrows():
        ab_w2v = w2v_of_abstract(paper[col_name], w2v)
        X_w2v[idx] = ab_w2v

    return X_w2v

# python 3

import numpy as np
import pandas as pd

def split_arxiv_by_code(arxiv, codes):
    """Return array of dataframes each containing unique code.

    Args:
        arxiv: DataFrame, of `abstracts` and `codes`
        codes: dict, mapping a code to its index

    Return:
        array, of DataFrames at index i containing only those abstracts of
            corresponding to code with index i.
    """
    return [arxiv[arxiv['Code'] == code] for code in codes]

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

def k_means_data_as_w2v_from_df(df, w2v, col_name = 'Abstract'):
    """Return array of training data for K-means consisting of w2v embeddings.

    Args:
        df: dataframe, containing column `col_name`
        w2v: dict, mapping words to w2v embeddings
        col_name: str, name of column in `df` containing abstracts

    Return:
        array: of shape (len(df), dim(w2v)).
    """

    # reset index on df
    df_ = df.reset_index(drop=True)

    assert (col_name in df.columns)

    # for K-means in scikit-learn the data should be
    # of the shape (length_of_data, dim_of_vector_space)
    n_papers = len(df_)
    dim_w2v = _compute_dim_w2v(w2v)
    X_w2v = np.zeros((n_papers, dim_w2v))

    # populate X with one-hot representatives
    for idx, paper in df_.iterrows():
        ab_w2v = w2v_of_abstract(paper[col_name], w2v)
        X_w2v[idx] = ab_w2v

    return X_w2v

def _compute_cluster_matrix(df, col1, col2, n_clusters):
    """Return matrix of pairwise counts of clusters.

    Args:
        df: DataFrame
        col1: str, e.g. 'lsa_pred'
        col2: str, e.g. 'tfidf_pred'
        n_clusters: int, number of clusters predicted
    """
    A = np.zeros((n_clusters, n_clusters))
    for _, paper in df.iterrows():
        id1 = paper[col1]
        id2 = paper[col2]
        A[id1][id2] += 1
    return A

# NB. NOT USING THIS
def uniform_freq_dist(df, codes, col_name='Codes'):
    """Return a dataframe with uniform frequency distribution in `col_name`.
    """
    df_by_codes = [df[df[col_name] == code] for code in codes]
    min_count = np.min(list(map(len, df_by_codes)))

    df_unif = pd.DataFrame(columns=df.columns.tolist())
    for code in codes:
        idx = codes[code]
        df_unif = df_unif.append( df_by_codes[idx].sample(min_count) )

    return df_unif

# python 3
"""
We compare a variety of document-level embeddings of scientific texts.

- one-hot
- word-to-vec
- TFIDF
- LSA
- hashing

Our data-set is abstracts of papers from 'arXiv.math' together with their MSC
codes (a standard subject classification in mathematics).
"""

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn import metrics

from utils import *

arxiv = pd.read_pickle('data/arxiv.pkl')
ab_list = arxiv['Abstract'].tolist()
codes = pd.read_pickle('data/msc_to_index.pkl')
w2v = pd.read_pickle('data/w2v_dict_50.pkl')

if __name__ == '__main__':

    print('Data looks like this:')
    print(arxiv.head())
    print()
    print('Number of papers:', len(arxiv))

    # create vocab
    vocab = build_vocab_from_corpus(ab_list)
    n_words = len(vocab)
    print('Number of words in vocab is', n_words)
    print()

    # create labels
    labels = np.asarray([codes[x] for x in arxiv['Code'].tolist()])

    # one-hot
    print('Building one-hot features...')
    X_oh = k_means_data_as_one_hot_from_df(arxiv, vocab)

    # w2v
    print('Building word-to-vec features...')
    X_w2v = k_means_data_as_w2v_from_abstracts(ab_list, w2v)
    normalizer = Normalizer(copy=False)
    X_w2v = normalizer.transform(X_w2v)

    # tfidf
    print('Building TFIDF features...')
    max_features = 10000
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=max_features,
                                 min_df=2, stop_words='english',
                                 use_idf=True)

    abs_corpus = [' '.join(ab) for ab in ab_list]
    X_tfidf = vectorizer.fit_transform(abs_corpus)

    # lsa
    print('Building LSA features...')
    n_lsa = 20
    svd = TruncatedSVD(n_lsa)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    X_lsa = lsa.fit_transform(X_tfidf)

    # hashing
    print('Building hashing features...')
    n_features = 2**18 # recommended by scikit-learn
    hasher = HashingVectorizer(n_features=n_features,
                               stop_words='english',
                               alternate_sign=False, norm='l2',
                               binary=False)

    X_hash = hasher.fit_transform(abs_corpus)

    # compare methods

    X_methods = [X_oh, X_w2v, X_tfidf, X_lsa, X_hash]
    method_name = ['one_hot', 'word_to_vec', 'tfidf', 'lsa', 'hasher']

    print()
    print('Compare different embeddings.')
    for idx, X in enumerate(X_methods):
        if method_name[idx] != 'lsa':
            km = MiniBatchKMeans(n_clusters=len(codes))
        else:
            km = KMeans(n_clusters=len(codes))

        print('Fitting K-Means to data with %s data' %method_name[idx])
        km.fit(X)

        # evaluate
        v_measure = metrics.v_measure_score(labels, km.labels_)

        print("Complete. V-measure: %0.3f" %v_measure)
        print()
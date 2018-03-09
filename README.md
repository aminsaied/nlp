# NLP
A (growing) collection of projects, experiments and examples relating to NLP.

## A Comparison of Document-embeddings for Scientific Text
A demonstration of several document-embedding methods as applied to data collected from [arXiv.math](https://arxiv.org/archive/math). Run the script with:

~~~~
$ python3 comparison_of_document_embeddings.py
~~~~

In particular, the following techniques are demoed:

- One-hot embedding
- Word-to-vec (w2v)
- TFIDF Analysis
- Latent Semantic Analysis (LSA)
- Hashing

We use the *V-measure* to compare these embedding techniques. Building the following visualisations can help interpret the performance of the models.

<img src="images/lsa_w2v_tfidf_.png" alt="Drawing" style="width: 1000px;"/>

Here the `(i,j)` entry counts the number of papers in cluster-`i` and with code-`j`.

<!-- These methods are fairly mainstream, and can all be implemented with Scikit-learn, for example.  

The aim of this demonstration is to compare the efficacy of these embeddings with regard to a clustering problem; namely text classification of scientific texts.

Novel ideas:

- Automatically assign subjects as clusters in the embedded data
- Ensemble method: use two or more of the most effective embeddings to find *cluster pairs* -->

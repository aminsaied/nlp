# A Comparison of Document-embeddings for Scientific Text
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

<img src="comparison_of_document_embeddings_for_scientific_text/images/lsa_w2v_tfidf_.png" alt="Drawing" style="width: 1000px;"/>

Here the `(i,j)` entry counts the number of papers in cluster-`i` and with code-`j`.

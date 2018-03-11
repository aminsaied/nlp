# NLP
A (growing) collection of projects, experiments and examples relating to NLP.

## A Comparison of Document Embeddings for Scientific Text

We compare a variety of document embedding techniques on a corpus of mathematics papers from the [arXiv](https://arxiv.org/archive/math). The general problem is to represent texts as vectors in some meaningful way (relationships between the texts are captured as relationships between their corresponding vectors).

In particular, the following techniques are compared:

- One-hot embedding
- Word-to-vec (w2v)
- TFIDF Analysis
- Latent Semantic Analysis (LSA)
- Hashing

### Detecting clusters in the space of embedded documents

We compare the effectiveness of the given embeddings as follows. The arXiv papers in our dataset are labelled with mathematics subject classification codes (MSCs). After we embed the papers in their respective vector spaces, we run **k-means** to detect clusters in the data. We can then compare the cluster labels (y_hat) with the MSC labels (y); There is a numeric comparison called the **V-measure**, or for the more visually inclined we have the following:

<img src="comparison_of_document_embeddings_for_scientific_text/images/lsa_w2v_tfidf_.png" alt="Drawing" style="width: 1000px;"/>

Here the `(i,j)` entry counts the number of papers in cluster-`i` and with code-`j`.

For a more thorough description see my [webpage](https://aminsaied.github.io/attachments/comparison_of_doc_embeddings/comparison_of_document_level_embeddings.slides.html).

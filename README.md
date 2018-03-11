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

## Building a Deep RNN with Bidirectional GRUs in Keras

The purpose of this script is two-fold:

1. To build a light-weight version of our [Cornetto](https://github.com/aminsaied/cornetto) library for demonstration purposes.
2. As an excuse to build an RNN with Keras.

[Cornetto](https://github.com/aminsaied/cornetto) was designed to classify mathematics papers based on their abstracts. Its central model is of the same design as the model we construct here and was implemented in TensorFlow.

Here's a sketch of the model we'll build. It is essentially identical to the `Cornetto` model.

<img src="images/model_design.png" style="width: 750px;"/>

where:

- $x^{\langle i \rangle}$ is the $i$-th input: for us, the $i$-th word in the abstract
- GRU cells allow for long-range dependencies and are faster to train then LSTM cells
- the orange arrows pass a hidden state from left-to-right
- the green arrows pass a hidden state from right-to-left (this is the bidirectional part!)


It turns out I'm a big fan of Keras! As you'll see here, we can get a fairly intricate network up and running very quickly and with minimal fuss. In just 7 lines of code, and with a small dataset we get an accuracy of ~80-85%.

I go into more detail on my [webpage](https://aminsaied.github.io/attachments/rnn_keras/rnn_keras.slides.html).

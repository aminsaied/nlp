# Tictoc Text Clustering Algorithm

This is an experimental algorithm designed to identify subjects in scientific text. It has two alternating steps - a tic-step and a toc-step.

## Tic Step
- Embed documents with _fixed_ embedding matrix `E`
- Run K-means (or EM algorithm) to identify clusters labels `y`
- Pass cluster labels to toc-step

## Toc Step
- Train an RNN with an embedding layer `E` to predict _fixed_ cluster labels `y`
- Update embedding layer as part of training (via backprop)
- Pass updated embedding matrix `E` to tic-step

Here is an overview of the algorithm.

<img src="tictoc_text_clustering/images/tictoc_overview.png" alt="Drawing" style="width: 1000px;"/>

We set up a small experiment:

~~~~
python3 tictoc_experiment.py
~~~~

It seems to show improvement in both the V-measure of the clusters and in the accuracy of the RNN both in the order of 1-2%. Caveat: one small experiment, tiny dataset. But at least it didn't just collapse to something trivial :-)

Next steps:
- examine the changes to the word embeddings: how dramatically are word-vectors changing?
- examine changes to the clusters: how dynamic are the changes to the clusters?

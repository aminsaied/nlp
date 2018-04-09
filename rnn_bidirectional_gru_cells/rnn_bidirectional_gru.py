#!/usr/bin/env python3
"""Script training bidirectional RNN with GRU cells on arXiv data in Keras.

Uses Keras to build an RNN with two layers of bidirectional GRU cells and a
softmax classifier. The model takes as input abstracts from the arXiv and
outputs their (Primary) MSC code.

This model is trained on a relatively small dataset consisting of ~10,000
samples.
"""
import os

from keras.models import Model
from keras.layers import Dense, Input, Bidirectional, GRU
from keras.layers.embeddings import Embedding
from keras.initializers import glorot_uniform

from utils import arxiv_dataset

# surpress some tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# set hyperparameters
max_len = 50
n_hidden = 128
p_drop = 0.7
batch_size = 256
n_epochs = 5

if __name__ == "__main__":

    print('Loading data...')
    N_TEST = 2000
    dataset = arxiv_dataset(max_len, N_TEST)
    arxiv, emb_matrix, training_data, test_data = dataset

    # see the training/test data
    X_train, Y_train = training_data
    X_test, Y_test   = test_data
    print('X_train shape:', X_train.shape, 'Y_train shape:', Y_train.shape)
    print('X_test shape:', X_test.shape, 'Y_test shape:', Y_test.shape)
    m, K = Y_train.shape

    vocab_len, emb_dim = emb_matrix.shape
    print('Size of vocab:', vocab_len)
    print('Word embeddings have dimension %s.\n' %emb_dim)

    print('Take a look at the arxiv data')
    print(arxiv.head(),'\n')

    # Build embedding layer
    embedding_layer = Embedding(input_dim = vocab_len,
                                output_dim = emb_dim,
                                trainable = False)
    embedding_layer.build((None,))
    embedding_layer.set_weights([emb_matrix])

    print('Building the model...')
    sentence_indices = Input(shape=(max_len,), dtype='int32')

    E = embedding_layer(sentence_indices)
    X = Bidirectional(GRU(n_hidden,
                          dropout=p_drop,
                          recurrent_initializer=glorot_uniform(),
                          return_sequences=True))(E)
    X = Bidirectional(GRU(n_hidden,
                          dropout=p_drop,
                          recurrent_initializer=glorot_uniform()))(X)
    X = Dense(units=K, activation='softmax')(X)

    model = Model(inputs=sentence_indices, outputs=X)

    print(model.summary())

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print('Training the model...')
    model.fit(X_train, Y_train,
              epochs=n_epochs,
              batch_size=batch_size)

    print('Evaluating the model...')
    loss, acc = model.evaluate(X_test, Y_test)
    print("Test accuracy = ", acc)

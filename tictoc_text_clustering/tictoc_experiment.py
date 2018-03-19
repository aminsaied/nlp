# python 3

from tictoc import Tic, Toc
from utils import *

# useful parameters
K      = 9     # number of classes
N_TEST = 2000  # number of test examples
N_ITER = 5     # number of tic-toc iterations

if __name__ == '__main__':

    # load dataset
    arxiv, emb_matrix, X_indices, Y_labels = load_data()

    # set up tic-step
    tic = Tic(arxiv, emb_matrix, X_indices, Y_labels)

    # set up toc-step
    toc = Toc(X_indices, emb_matrix, n_classes=K, n_test=N_TEST)

    for iter_ in range(N_ITER):
        print("Starting iteration %s\n"%iter_)

        # tic-step
        tic.set_embedding_matrix(emb_matrix)
        arxiv = tic.update_clusters()
        print( "V_measure: {0:.4f} \n".format(tic.v_measure_history[-1]) )

        # toc-step
        toc.set_clusters(arxiv)
        emb_matrix = toc.update_embedding_matrix()
        print( "\nAccuracy: {0:.4f} \n".format(toc.accuracy_history[-1]) )

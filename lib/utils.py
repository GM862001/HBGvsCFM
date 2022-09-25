import numpy as np


def compute_best_loss_k(loss):

    best_loss_k = []
    best_loss = np.infty
    for i, l in enumerate(loss):
        if l < best_loss:
            best_loss = l
        best_loss_k.append(best_loss)
    return best_loss_k

import numpy as np
from scipy.special import comb as choose

def bezier_matrix(degree):
    m = degree
    Q = np.zeros((degree + 1, degree + 1))
    for i in range(degree + 1):
        for j in range(degree + 1):
            if (0 <= (i+j)) and ((i+j) <= degree):
                Q[i,j] = choose(m, j) * choose(m-j, m-i-j) * ((-1)**(m-i-j))
    return Q
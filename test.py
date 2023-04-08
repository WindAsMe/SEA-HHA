import numpy as np
from copy import deepcopy


def diag_multi(matrix, vector):
     r, c = matrix.shape
     for i in range(r):
          for j in range(c):
               matrix[i][j] *= vector[j]
     return matrix


def self_multi(vector):
     t_vector = deepcopy(vector)
     for i in range(len(t_vector)):
          t_vector[i] *= t_vector[i]
     return np.array(t_vector)


def binary_random(m, M):
     """
     :param m: scale of phi
     :param M: budget
     :return:
     """
     tM = np.random.choice(list(range(0, M+1)))

     phi = np.zeros(m)
     index = np.random.choice(list(range(0, m)), tM)
     for i in index:
          phi[i] = 1
     return phi


def isNegative(vector):
     for e in vector:
          if e > 0:
               return False
     return True


def cons(phi, b, H):
     """
     :param phi: binary decision variables
     :param b: continuous decision variables
     :param H: relationship matrix
     :return:
     """
     n, m = H.shape
     R = np.ones(n)
     f = list(range(m, 0, -1))

     H_hat = diag_multi(diag_multi(H, phi), f)
     b_hat = self_multi(b).reshape(-1, 1)

     temp = np.dot(H_hat, b_hat)[:, 0]
     constraints = R - temp
     return constraints


if __name__ == "__main__":
     H = np.array([[1, 1, 0, 0, 1, 1],
                   [0, 0, 1, 0, 1, 0],
                   [0, 1, 0, 1, 0, 1],
                   [0, 0, 1, 1, 1, 1]])
     n, m = H.shape
     M = 2

     for i in range(10000):
          phi = binary_random(m, M)
          b = np.random.random(m)
          # phi = [0, 1, 1, 0, 0, 0]
          # b = [0.00000032, 0.447215507501615, 0.500001607984798, 0.00000032, 0.00000032, 0.00000032]
          constraints = cons(phi, b, H)
          if isNegative(constraints):
               print("optimum: ", sum(b))

















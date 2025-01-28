import numpy as np
import matplotlib.pyplot as plt
import itertools
import time
from graphs import *

n = 3
m = 1
eta = 1
samples = 2

h = 3
disagreement = []


def sample_binary_arrays_large_N(N, K):
    sampled = set()
    result = []
    while len(result) < K:
        arr = np.random.choice([-1, 1], size=N)
        if arr[0] == -1:
            arr = -arr
        arr_tuple = tuple(arr)
        if arr_tuple not in sampled:
            sampled.add(arr_tuple)
            result.append(arr)
        if np.log2(len(sampled)) >= N - 1:
            break
    return np.array(result)


# Adjacency matrix
G = small_line

G = np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]])

G = G / np.sum(G, axis=1)[:, np.newaxis]

print(G)

x_L = np.zeros((n - m, 1))

u = np.ones((m * (h + 1), 1))

W = G

W_LL = W[0 : (n - m), 0 : (n - m)]
W_LM = W[0 : (n - m), (n - m) :]


W_LL_H = np.linalg.matrix_power(W_LL, h)

A = np.zeros((n - m, m))

for t in range(h):
    A = np.hstack((np.linalg.matrix_power(W_LL, t) @ W_LM, A))

print(A @ u)
exit()

P = np.vstack((np.eye(n - m), np.zeros((m, n - m))))
D = 1 / n * np.eye(n) - 1 / (n**2) * np.ones((n, n))

import numpy as np
import matplotlib.pyplot as plt
import itertools
import time
from graphs import *

m = 1
eta = 100
samples = 2**20

h = 20
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
G = long_line
n = G.shape[0]
W = G / np.sum(G, axis=1)[:, np.newaxis]

x_L = np.zeros((n - m, 1))

W_LL = W[0 : (n - m), 0 : (n - m)]
W_LM = W[0 : (n - m), (n - m) :]
W_LL_H = np.linalg.matrix_power(W_LL, h)

A = np.zeros((n - m, m))

for t in range(h):
    A = np.hstack((np.linalg.matrix_power(W_LL, t) @ W_LM, A))

P = np.vstack((np.eye(n - m), np.zeros((m, n - m))))
D = 1 / n * np.eye(n) - 1 / (n**2) * np.ones((n, n))


start_time = time.time()
print("H: ", h)

Q = np.vstack(
    (np.zeros((n - m, m * (h + 1))), np.hstack((np.zeros((m, m * h)), np.eye(m))))
)

U_Cast = P @ A + Q
x_Cast = P @ W_LL_H @ x_L

S = U_Cast.T @ D @ U_Cast
K = 2 * x_Cast.T @ D @ U_Cast
c = x_Cast.T @ D @ x_Cast


def dis(u):
    return u.T @ S @ u + K @ u + c


# Search for the optimal u

rand_u_vec = sample_binary_arrays_large_N(m * (h + 1), samples) * eta
best_u = rand_u_vec[0]

for i in range(rand_u_vec.shape[0]):
    rand_u = rand_u_vec[i]
    if dis(rand_u) > dis(best_u):
        best_u = rand_u
print(f"Disagreement: {dis(best_u)}")
disagreement.append(dis(best_u[h])[0][0])

print("--- %s seconds ---" % (time.time() - start_time))


rand_u_vec = sample_binary_arrays_large_N(m * (h + 1), samples) * eta
best_val = 0
print("All Optimal Vectors:")
optimal_vectors = []
for vec in rand_u_vec:
    if dis(vec) >= best_val:
        # print(vec)
        optimal_vectors.append(vec)


# Plot red box for -eta, blue box for +eta, in a line
optimal_vectors = optimal_vectors[0:10]
fig, ax = plt.subplots()
ax.set_xlim(0, m * (max_h + 1))
ax.set_ylim(0, len(optimal_vectors))
ax.set_aspect("equal")

for i, optimal_vector in enumerate(optimal_vectors):
    for j in range(len(optimal_vector)):
        if optimal_vector[j] < 0:
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color="red"))
        else:
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color="blue"))

plt.show()

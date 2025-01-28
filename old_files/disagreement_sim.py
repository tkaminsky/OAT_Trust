import numpy as np
import matplotlib.pyplot as plt
import itertools
import time
from graphs import *


# Helper function samples K binary arrays of length N without replacement
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


# Vector of horizons to consider
h_vals = range(1, 20)
max_h = max(h_vals)

# Number of malicious nodes
m = 1
# Maximum value of the adversarial input
eta = 1
# Number of samples (for approximating optimum)
samples = 2**20

# This will store the optimal control inputs for each h, and the corresponding disagreement
best_u = {}
disagreement = []


# Adjacency matrix
G = small_line
n = G.shape[0]
W = G / np.sum(G, axis=1)[:, np.newaxis]

x_L = np.zeros((n - m, 1))

W_LL = W[0 : (n - m), 0 : (n - m)]
W_LM = W[0 : (n - m), (n - m) :]
W_LL_H_vals = [np.linalg.matrix_power(W_LL, h) for h in h_vals]

A_full = np.zeros((n - m, m))

for t in range(max_h):
    A_full = np.hstack((np.linalg.matrix_power(W_LL, t) @ W_LM, A_full))

P = np.vstack((np.eye(n - m), np.zeros((m, n - m))))
D = 1 / n * np.eye(n) - 1 / (n**2) * np.ones((n, n))


for h in h_vals:
    start_time = time.time()
    print("H: ", h)
    Q = np.vstack(
        (np.zeros((n - m, m * (h + 1))), np.hstack((np.zeros((m, m * h)), np.eye(m))))
    )
    A = A_full[:, (-m * (h + 1)) :]
    W_LL_H = W_LL_H_vals[h_vals.index(h)]

    U_Cast = P @ A + Q

    x_Cast = P @ W_LL_H @ x_L

    S = U_Cast.T @ D @ U_Cast
    K = 2 * x_Cast.T @ D @ U_Cast
    c = x_Cast.T @ D @ x_Cast

    def dis(u):
        return u.T @ S @ u + K @ u + c

    # Search for the optimal u

    rand_u_vec = sample_binary_arrays_large_N(m * (h + 1), samples) * eta
    best_u[h] = rand_u_vec[0]

    for i in range(rand_u_vec.shape[0]):
        rand_u = rand_u_vec[i]
        if dis(rand_u) > dis(best_u[h]):
            best_u[h] = rand_u
    print(f"Disagreement: {dis(best_u[h])}")
    disagreement.append(dis(best_u[h])[0][0])

    print("--- %s seconds ---" % (time.time() - start_time))

# Plot red box for -eta, blue box for +eta, in a line
fig, ax = plt.subplots()
ax.set_xlim(0, m * (max_h + 1))
ax.set_ylim(0, len(h_vals))
ax.set_aspect("equal")
for i, h in enumerate(h_vals):
    for j in range(h + 1):
        if best_u[h][j] < 0:
            ax.add_patch(plt.Rectangle((j * m, i), 1, 1, color="red"))
        else:
            ax.add_patch(plt.Rectangle((j * m, i), 1, 1, color="blue"))


# Label the x-axis 'time'
plt.xlabel("Time")
# Label y-axis 'Time Horizon (h)'
plt.ylabel("Time Horizon (H)")
plt.title("Optimal Adversarial Input vs Time Horizon")
# Add a dotted line at log_2(samples)
plt.axhline(y=np.log2(samples), color="green", linestyle="--")

plt.show()

# Plot disagreement
plt.plot(h_vals, disagreement)
plt.xlabel("Time Horizon (H)")
plt.ylabel("Disagreement")
plt.title("Maximum Disagreement vs Time Horizon")

plt.show()

disagree_config = (P @ A + Q) @ best_u[max_h] + P @ W_LL_H @ x_L
print("Best-case final values: ", disagree_config)

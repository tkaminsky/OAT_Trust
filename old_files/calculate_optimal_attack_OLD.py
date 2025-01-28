import numpy as np
import matplotlib.pyplot as plt
import itertools
import time
from graphs import *
import argparse
import yaml
import os
from samplers import k_block_sampler

# Read -h argument for the graph
parser = argparse.ArgumentParser()
parser.add_argument("-H", "--horizon", help="Horizon", type=int, default=None)
# Add -p argument for the path to the config file
parser.add_argument(
    "-c",
    "--config",
    help="Path to the config file",
    type=str,
    default="./base_config.yaml",
)
args = parser.parse_args()

# Read the config file
with open(args.config, "r") as file:
    config = yaml.safe_load(file)

# Read graph or throw error if not in config
if "graph" in config:
    graph = config["graph"]
    G = graph_dict[graph]
else:
    raise ValueError("Graph not found in config file")

# Read m
if "m" in config:
    m = config["m"]
else:
    raise ValueError("m not found in config file")

# Read eta
if "eta" in config:
    eta = config["eta"]
else:
    raise ValueError("eta not found in config file")

# Read samples
if "samples_log_2" in config:
    samples = 2 ** config["samples_log_2"]
else:
    raise ValueError("samples not found in config file")

if "enforce_symmetry" in config:
    enforce_symmetry = config["enforce_symmetry"]

# Read the horizon from h argument or config file
if args.horizon is not None:
    h = args.horizon
elif "H" in config:
    h = config["H"]
else:
    raise ValueError("Horizon not found in config file or as argument")

if args.save is not None:
    save_file = args.save
elif "save_file" in config:
    save_file = config["save_file"]
else:
    raise ValueError("Save file not found in config file or as argument")

# Helper function samples K binary arrays of length N without replacement
sampler = k_block_sampler

n = G.shape[0]
W = G / np.sum(G, axis=1)[:, np.newaxis]

x_L = np.zeros((n - m, 1))


W_LL = W[0 : (n - m), 0 : (n - m)]
W_LM = W[0 : (n - m), (n - m) :]

A = np.zeros((n - m, m))

for t in range(h):
    A = np.hstack((np.linalg.matrix_power(W_LL, t) @ W_LM, A))

D = 1 / (n - m) * np.eye(n - m) - 1 / ((n - m) ** 2) * np.ones((n - m, n - m))

start_time = time.time()
print("H: ", h)

W_LL_H = np.linalg.matrix_power(W_LL, h)

U_Cast = A
x_Cast = W_LL_H @ x_L

S = U_Cast.T @ D @ U_Cast
K = 2 * x_Cast.T @ D @ U_Cast
c = x_Cast.T @ D @ x_Cast


def dis(u):
    return u.T @ S @ u + K @ u + c


# Search for the optimal u

k = 2

# rand_u_vec = sample_binary_arrays_large_N(m*(h+1), samples, enforce_net_symmetry=enforce_symmetry) * eta
rand_u_vec = sampler(m * (h + 1), k, enforce_symmetry) * eta
best_u = rand_u_vec[0]

for i in range(rand_u_vec.shape[0]):
    rand_u = rand_u_vec[i]
    if dis(rand_u) > dis(best_u):
        best_u = rand_u
print(f"Disagreement: {dis(best_u)}")
disagreement = dis(best_u)[0][0]


print("--- %s seconds ---" % (time.time() - start_time))


# Create save file if it doesn't exist (numpy dictionary)
save_file = "./" + save_file + ".npy"
if not os.path.exists(save_file):
    np.save(save_file, {})
# Load the save file
save_dict = np.load(save_file, allow_pickle=True).item()
# Add the results to the save file
save_dict[h] = {"best_u": best_u, "disagreement": disagreement}
# Save the save file
np.save(save_file, save_dict)

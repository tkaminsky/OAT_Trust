import numpy as np
import matplotlib.pyplot as plt
import itertools
import time
from graphs import *
import argparse
import yaml
import os
from samplers import k_block_sampler, sample_binary_arrays_large_N

x_init_options = ["zeros", "hot", "cold-hot", "alternating"]

# Action dict has keys 0,...,m-1 and each key is an h-length action sequence (per-agent actions)
def combine_malicious_actions(action_dict):
    m = len(action_dict.keys())
    h = len(action_dict[0])
    combined_actions = np.zeros((m * h, 1))
    for i in range(m):
        for j in range(h):
            combined_actions[i * h + j] = action_dict[i][j]
    return combined_actions


def save_results(root_dir, save_string, h, best_u, disagreement):
    # If root dir does not exist, create it
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    # Create save file if it doesn't exist (numpy dictionary)
    if not os.path.exists(save_string):
        np.save(save_string, {})

    # Load the save file
    save_dict = np.load(save_string, allow_pickle=True).item()
    # Add the results to the save file
    save_dict[h] = {"best_u": best_u, "disagreement": disagreement}
    # Save the save file
    np.save(save_string, save_dict)


def run_experiment(m, h, k, enforce_symmetry, eta, dis, sampling="k_block"):
    start_time = time.time()

    if sampling == "k_block":
        # Find all possible k-block actions over H timesteps
        u_vec = k_block_sampler((h + 1), k, enforce_symmetry) * eta
        n_actions = u_vec.shape[0]
    elif sampling == "random":
        samples = 2**20
        u_vec = (
            sample_binary_arrays_large_N(
                m * (h + 1), samples, enforce_net_symmetry=enforce_symmetry
            )
            * eta
        )

    mid_time = time.time()

    init_action_dict = {i: u_vec[0] for i in range(m)}

    best_u = combine_malicious_actions(init_action_dict)
    best_action_dict = init_action_dict

    for i in range(n_actions ** m):
        action_dict = {
            j: u_vec[(i // (n_actions ** j)) % n_actions] for j in range(m)
        }
        u_now = combine_malicious_actions(action_dict)
        if dis(u_now) > dis(best_u):
            best_u = u_now
            best_action_dict = action_dict


    print(f"Disagreement: {dis(best_u)}")
    disagreement = dis(best_u)[0][0]

    end_time = time.time()

    print(
        "--- %.2fs sample / %.2fs eval ---" % (mid_time - start_time, end_time - mid_time)
    )

    return best_u, best_action_dict, disagreement


def get_disagreement_fn(n, m, h, W, x_L):
    W_LL = W[0 : (n - m), 0 : (n - m)]
    W_LM = W[0 : (n - m), (n - m) :]

    A = np.zeros((n - m, m))

    for t in range(h):
        A = np.hstack((np.linalg.matrix_power(W_LL, t) @ W_LM, A))

    D = 1 / (n - m) * np.eye(n - m) - 1 / ((n - m) ** 2) * np.ones((n - m, n - m))

    print("H: ", h)

    W_LL_H = np.linalg.matrix_power(W_LL, h)

    U_Cast = A
    x_Cast = W_LL_H @ x_L

    S = U_Cast.T @ D @ U_Cast
    K = 2 * x_Cast.T @ D @ U_Cast
    c = x_Cast.T @ D @ x_Cast

    def dis(u):
        return u.T @ S @ u + K @ u + c

    return dis


def parse_params(config, max_H_input, horizon):
    # Read the config file
    with open(config, "r") as file:
        config = yaml.safe_load(file)

        if max_H_input is not None:
            max_H = str(max_H_input)
        else:
            max_H = "Unknown"

        if "k" in config:
            k = config["k"]
        else:
            k = 2

        # Read graph or throw error if not in config
        if "graph" in config:
            graph = config["graph"]
            G = get_graph(graph)
        else:
            raise ValueError("Graph not found in config file")

        # Read m
        if "m" in config:
            m = config["m"]
            if m > 1:
                # Move m//2 nodes from the beginning to the end of the graph
                G = np.roll(G, -m // 2, axis=0)
        else:
            raise ValueError("m not found in config file")

        # Read eta
        if "eta" in config:
            eta = config["eta"]
        else:
            raise ValueError("eta not found in config file")

        if "enforce_symmetry" in config:
            enforce_symmetry = config["enforce_symmetry"]

        # Read the horizon from h argument or config file
        if horizon is not None:
            h = horizon
        elif "H" in config:
            h = config["H"]
        else:
            raise ValueError("Horizon not found in config file or as argument")

        # Read root_dir from config file
        if "root_dir" in config:
            root_dir = config["root_dir"]
        else:
            root_dir = "./"

        # Get x_init from config file
        if "x_inits" in config:
            x_init = config["x_inits"]
        else:
            x_init = "zeros"

        if x_init not in x_init_options:
            raise ValueError(f"x_init must be one of {x_init_options}")

        save_string = os.path.join(
            root_dir,
            f"{graph}_H{max_H}_k{k}_x{x_init}_m{m}_eta{eta}_ES{1 if enforce_symmetry else 0}.npy",
        )

        n = G.shape[0]
        W = G / np.sum(G, axis=1)[:, np.newaxis]

        if x_init == "zeros":
            x_L = np.zeros((n - m, 1))
        elif x_init == "hot":
            x_L = np.ones((n - m, 1)) * eta
        elif x_init == "cold-hot":
            x_L = np.ones((n - m, 1)) * eta
            x_L[0 : ((n - m) // 2 + 1)] = -eta
        elif x_init == "alternating":
            x_L = np.ones((n - m, 1)) * eta
            x_L[0 : (n - m + 1) : 2] = -eta
        else:
            print("This should not happen")

        return (k, m, eta, enforce_symmetry, h, root_dir, save_string, n, W, x_L)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-H", "--horizon", help="Horizon", type=int, default=None)
    parser.add_argument("-c", "--config", help="Path to the config file", type=str, required=True)
    parser.add_argument("-m", "--max_H", help="Max horizon", type=int, default=None)
    args = parser.parse_args()

    # Get the parameters from the config file and arguments
    k, m, eta, enforce_symmetry, h, root_dir, save_string, n, W, x_L = parse_params(args.config, args.max_H, args.horizon)

    # Get the disagreement function based on graph and parameters
    dis = get_disagreement_fn(n, m, h, W, x_L)

    # Run the experiment
    best_u, best_action_dict, disagreement = run_experiment(m, h, k, enforce_symmetry, eta, dis)

    # Save the results
    save_results(root_dir, save_string, h, best_u, disagreement)


if __name__ == "__main__":
    main()

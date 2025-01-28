import numpy as np
import time


def get_all_binary_arrays(N, enforce_symmetry=False):
    """
    Get all binary arrays of length N, with +1 and -1 instead of 1 and 0
    """
    result = []

    # Base case
    if N == 1:
        return [[1], [-1]] if not enforce_symmetry else [[1]]

    # Recursive case
    for arr in get_all_binary_arrays(N - 1, enforce_symmetry=False):
        result.append([1] + arr)
        if not enforce_symmetry:
            result.append([-1] + arr)

    return result


def get_all_k_blocks(N, k):
    """
    Get all possible ways to split a list of size N k times
    """
    result = []

    # Base case
    if k == 1:
        return [[i] for i in range(1, N)]

    # Recursive case
    first_split = [[i] for i in range(1, N - k + 1)]
    for split in first_split:
        for block in get_all_k_blocks(N - split[0], k - 1):
            block_local = [i + split[0] for i in block]
            result.append(split + block_local)

    return result


def k_block_sampler(dimension, k=2, enforce_sym=False):
    """
    Gathers all possible k-blocks from a given dimension
    """
    if dimension == 1:
        return np.array([[1], [-1]])
    elif dimension == 2:
        return (
            np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
            if not enforce_sym
            else np.array([[1, 1], [1, -1]])
        )
    if dimension <= k:
        k = dimension - 1
        return k_block_sampler(dimension, k, enforce_sym)
    result = []
    k_cuts = get_all_k_blocks(dimension, k)
    signs = get_all_binary_arrays(k + 1, enforce_sym)

    # print(k_cuts)
    for cut in k_cuts:
        cut = [0] + cut + [dimension]
        for sign in signs:
            arr = np.ones(dimension)
            # Add signs to each block
            for i in range(k + 1):
                arr[cut[i] : cut[i + 1]] *= sign[i]

            if not any(np.array_equal(arr, x) for x in result):
                result.append(arr)

    return np.array(result)


def sample_binary_arrays_large_N(N, K, enforce_net_symmetry):
    sampled = set()
    result = []
    while len(result) < K:
        arr = np.random.choice([-1, 1], size=N)
        if enforce_net_symmetry and arr[0] == -1:
            arr = -arr
        arr_tuple = tuple(arr)
        if arr_tuple not in sampled:
            sampled.add(arr_tuple)
            result.append(arr)
        if np.log2(len(sampled)) >= N - 1:
            break
    return np.array(result)

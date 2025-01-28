import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# Accept the file path as an argument
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help="Path to the file", type=str, default=None)
args = parser.parse_args()

# Read the file path
file_path = args.file
file_path_prefix = file_path.rsplit(".npy", 1)[0]
# Remove any superdirectories
file_path_prefix = file_path_prefix.split("/")[-1]
subdir = file_path.split("/")[-2]

# Load the file
file = np.load(file_path, allow_pickle=True)

# Look for '_m' in the file name, and take m to be all the digits until the next _
m = int(file_path_prefix.split("_m")[1].split("_")[0])

disagreement_vec = []
best_u_vec = []
h_vals = []
for key in file.item():
    h_vals.append(key)
    best_u_vec.append(file.item()[key]["best_u"])
    disagreement_vec.append(file.item()[key]["disagreement"])

max_h = max(h_vals)
# m = 2

# Sort the h_vals and sort best_u_vec and disagreement_vec accordingly
h_vals, best_u_vec, disagreement_vec = zip(
    *sorted(zip(h_vals, best_u_vec, disagreement_vec))
)


# Plot red box for -eta, blue box for +eta, in a line
fig, ax = plt.subplots()
ax.set_xlim(0, m * (max_h))
ax.set_ylim(0, len(h_vals))
ax.set_aspect("equal")
for i, h in enumerate(h_vals):
    u_vec = best_u_vec[i]
    print(u_vec)
    for j in range(m * h):
        if u_vec[j] < 0:
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color="blue"))
        else:
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color="red"))

# Create media/subdir subdirectory if it doesn't exist
if not os.path.exists(f"media/{subdir}"):
    os.makedirs(f"media/{subdir}")

# Create subdir/{file_path_prefix} subdirectory if it doesn't exist
if not os.path.exists(f"media/{subdir}/{file_path_prefix}"):
    os.makedirs(f"media/{subdir}/{file_path_prefix}")


print(f"media/{subdir}/{file_path_prefix}/{file_path_prefix}_optimal_inputs.png")

# Save the first figure
plt.xlabel("Time")
plt.ylabel("Time Horizon (H)")
plt.title("Optimal Adversarial Input vs Time Horizon")
plt.savefig(f"media/{subdir}/{file_path_prefix}/{file_path_prefix}_optimal_inputs.png")
plt.close()

# Plot and save disagreement
plt.plot(h_vals, disagreement_vec)
plt.xlabel("Time Horizon (H)")
plt.ylabel("Disagreement")
plt.title("Maximum Disagreement vs Time Horizon")
plt.savefig(
    f"media/{subdir}/{file_path_prefix}/{file_path_prefix}_maximum_disagreement.png"
)
plt.close()

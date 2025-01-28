import numpy as np
import matplotlib.pyplot as plt

file_path = "./block_experiments/barbell_6_H20_xzeros_m1_eta1_ES0.npy"

# Load the file
file = np.load(file_path, allow_pickle=True)

disagreement_vec = []
best_u_vec = []
h_vals = []
for key in file.item():
    h_vals.append(key)
    best_u_vec.append(file.item()[key]["best_u"])
    disagreement_vec.append(file.item()[key]["disagreement"])

max_h = max(h_vals)
m = 1
samples = 2**21

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
    print(f"On {h}")
    print(u_vec)
    for j in range(h):
        if u_vec[j] < 0:
            ax.add_patch(plt.Rectangle((j * m, i), 1, 1, color="blue"))
        else:
            ax.add_patch(plt.Rectangle((j * m, i), 1, 1, color="red"))


# Label the x-axis 'time'
plt.xlabel("Time")
# Label y-axis 'Time Horizon (h)'
plt.ylabel("Time Horizon (H)")
plt.title("Optimal Adversarial Input vs Time Horizon")
# Add a dotted line at log_2(samples)
plt.axhline(y=np.log2(samples), color="green", linestyle="--")

plt.show()

# Plot disagreement
plt.plot(h_vals, disagreement_vec)
plt.xlabel("Time Horizon (H)")
plt.ylabel("Disagreement")
plt.title("Maximum Disagreement vs Time Horizon")

plt.show()

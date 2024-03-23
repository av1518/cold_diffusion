# %%
import json
import matplotlib.pyplot as plt

# Paths to your JSON files
cosine_json_file_path = "../metrics/fid_gaussian_cosine.json"
linear_json_file_path = "../metrics/fid_gaussian_linear.json"

zoom_5x5_set_path = "../metrics/fid_zoom_5x5_set.json"
zoom_distr_path = "../metrics/fid_zoom_distr.json"


# Function to load and extract data from JSON file
def load_fid_scores(file_path):
    with open(file_path, "r") as file:
        fid_scores = json.load(file)
    epochs = [int(epoch.split()[1]) for epoch in fid_scores]
    scores = [score for score in fid_scores.values()]
    return epochs, scores


# Load and extract data for both files
cosine_epochs, cosine_scores = load_fid_scores(cosine_json_file_path)
linear_epochs, linear_scores = load_fid_scores(linear_json_file_path)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(cosine_epochs, cosine_scores, marker="o", label="Cosine Schedule")
plt.plot(linear_epochs, linear_scores, marker="o", label="Linear Schedule")
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("FID", fontsize=12)
plt.grid(True)
plt.legend()
plt.savefig("../figures/fid_linear_cosine_comparison.png", dpi=300, bbox_inches="tight")
plt.show()
# %%
# Load and extract data for both files
zoom_5x5_epochs, zoom_5x5_scores = load_fid_scores(zoom_5x5_set_path)
zoom_distr_epochs, zoom_distr_scores = load_fid_scores(zoom_distr_path)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(
    zoom_5x5_epochs, zoom_5x5_scores, marker="o", label="Bilinear Zoom Set Sampling"
)
plt.plot(
    zoom_distr_epochs,
    zoom_distr_scores,
    marker="o",
    label="Nearest Interpolation Zoom Sampling",
)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("FID", fontsize=12)
plt.grid(True)
plt.legend()
plt.show()

# %%
# plot linear vs Nearest Interpolation Zoom Sampling
plt.figure(figsize=(8, 6))
plt.plot(linear_epochs, linear_scores, marker="o", label="Linear Schedule")
plt.plot(
    zoom_distr_epochs,
    zoom_distr_scores,
    marker="o",
    label="Nearest Interpolation Zoom Sampling",
)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("FID", fontsize=12)
plt.grid(True)
plt.legend()

# %% Plot everything together
plt.figure(figsize=(8, 6))
plt.plot(cosine_epochs, cosine_scores, marker="o", label="Cosine Schedule")
plt.plot(linear_epochs, linear_scores, marker="o", label="Linear Schedule")
plt.plot(
    zoom_5x5_epochs, zoom_5x5_scores, marker="o", label="Bilinear Zoom Set Sampling"
)
plt.plot(
    zoom_distr_epochs,
    zoom_distr_scores,
    marker="o",
    label="Nearest Interpolation Zoom Sampling",
)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("FID", fontsize=12)
plt.grid(True)
plt.legend()
plt.show()

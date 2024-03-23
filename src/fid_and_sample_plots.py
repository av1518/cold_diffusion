# %%
import json
import matplotlib.pyplot as plt
import torch
from torchvision.utils import save_image, make_grid
from strat_funcs import generate_4x4_z_T, single_alternating_zoom_batch

# %%
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

# %% Plot good and bad samples

linear_good = torch.load("./sample_sets/samples_gaussian_linear_good.pt")
linear_bad = torch.load("./sample_sets/samples_gaussian_linear_bad.pt")
cosine_good = torch.load("./sample_sets/samples_gaussian_cosine_good.pt")
cosine_bad = torch.load("./sample_sets/samples_gaussian_cosine_bad.pt")
zoom_5x5_good = torch.load("./sample_sets/samples_zoom_5x5_good.pt")
zoom_5x5_bad = torch.load("./sample_sets/samples_zoom_5x5.pt")
zoom_4x4_good = torch.load("./sample_sets/samples_zoom_4x4_good.pt")
zoom_4x4_bad = torch.load("./sample_sets/samples_zoom_4x4.pt")


def extract_samples(sample_tensor):
    samples = []
    for i in range(len(sample_tensor)):
        samples.append(sample_tensor[i][0].squeeze())
    return samples


linear_good = extract_samples(linear_good)
linear_bad = extract_samples(linear_bad)
cosine_good = extract_samples(cosine_good)
cosine_bad = extract_samples(cosine_bad)
zoom_5x5_good = extract_samples(zoom_5x5_good)
zoom_5x5_bad = extract_samples(zoom_5x5_bad)
zoom_4x4_good = extract_samples(zoom_4x4_good)
zoom_4x4_bad = extract_samples(zoom_4x4_bad)


def plot_samples(sample_list, title):
    # Stack the samples to create a single tensor
    samples_tensor = torch.stack(sample_list).unsqueeze(1)
    print(samples_tensor.shape)

    # Move the tensor to CPU
    samples_tensor_cpu = samples_tensor.cpu()

    # Create a grid of images
    grid = make_grid(samples_tensor_cpu, nrow=3, normalize=True)

    # Plot
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0))
    plt.title(title)
    plt.axis("off")
    plt.savefig(f"../figures/{title}.png", dpi=500, bbox_inches="tight")
    plt.show()


# %%
plot_samples(linear_good, "Guassian Noise (Linear Schedule) Good Samples")
plot_samples(linear_bad, "Gaussian Noise (Linear Schedule) Bad Samples")
plot_samples(cosine_good, "Gaussian Noise (Cosine Schedule) Good Samples")
plot_samples(cosine_bad, "Gaussian Noise (Cosine Schedule) Bad Samples")
plot_samples(zoom_5x5_good, "Zoom Bilinear Set Good Samples")
plot_samples(zoom_5x5_bad, "Zoom Bilinear Set Bad Samples")
plot_samples(zoom_4x4_good, "Zoom Nearest Distribution Good Samples")
plot_samples(zoom_4x4_bad, "Zoom Nearest Distribution Samples")

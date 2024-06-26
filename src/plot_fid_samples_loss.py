## @file plot_fid_samples_loss.py
#  @brief This script loads and plots the FID scores for all models, the good and bad samples saved
#  and the loss curves for all models.

# %%
import json
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

# %% Plot fid for linear and cosine schedules DDPM models
# Paths to JSON files
cosine_json_file_path = "../metrics/fid_gaussian_cosine.json"
linear_json_file_path = "../metrics/fid_gaussian_linear.json"

zoom_5x5_set_path = "../metrics/fid_zoom_5x5_set.json"
zoom_distr_path = "../metrics/fid_zoom_4x4_distr.json"


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
plt.plot(linear_epochs, linear_scores, marker="o", label="Linear Schedule")
plt.plot(cosine_epochs, cosine_scores, marker="o", label="Cosine Schedule")
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("FID", fontsize=12)
plt.legend()
plt.savefig("../figures/fid_linear_cosine_comparison.png", dpi=300, bbox_inches="tight")
# plt.show()
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
# plt.show()

# %% Plot linear vs Nearest Interpolation Zoom Sampling
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
plt.plot(linear_epochs, linear_scores, marker="o", label="DDPM Linear Schedule")
plt.plot(cosine_epochs, cosine_scores, marker="o", label="DDPM Cosine Schedule")
# plt.plot(
#     zoom_5x5_epochs, zoom_5x5_scores, marker="o", label="Bilinear Zoom Set Sampling"
# )
plt.plot(
    zoom_distr_epochs,
    zoom_distr_scores,
    marker="o",
    label="Cold Zoom Diffusion",
    color="green",
)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("FID", fontsize=12)
plt.legend()
plt.savefig("../figures/fid_all_models.png", dpi=300, bbox_inches="tight")
# plt.show()

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
    # Move the tensor to CPU
    samples_tensor_cpu = samples_tensor.cpu()

    # Create a grid of images
    grid = make_grid(
        samples_tensor_cpu, nrow=3, value_range=(-0.5, 0.5), normalize=True
    )

    # Plot
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.savefig(f"../figures/{title}.png", dpi=500, bbox_inches="tight")
    print(f"Saved {title}.png in figures directory")
    # plt.show()


# %%
plot_samples(linear_good, "Guassian Noise (Linear Schedule) Good Samples")
plot_samples(linear_bad, "Gaussian Noise (Linear Schedule) Bad Samples")
plot_samples(cosine_good, "Gaussian Noise (Cosine Schedule) Good Samples")
plot_samples(cosine_bad, "Gaussian Noise (Cosine Schedule) Bad Samples")
plot_samples(zoom_5x5_good, "Zoom Bilinear Set Good Samples")
plot_samples(zoom_5x5_bad, "Zoom Bilinear Set Bad Samples")
plot_samples(zoom_4x4_good, "Zoom Nearest Distribution Good Samples")
plot_samples(zoom_4x4_bad, "Zoom Nearest Distribution Samples")

# %% Plot the loss curves

linear_loss_path = "../metrics/losses_ddpm_linear.json"
cosine_loss_path = "../metrics/losses_ddpm_cosine.json"

with open(linear_loss_path, "r") as file:
    linear_loss = json.load(file)
with open(cosine_loss_path, "r") as file:
    cosine_loss = json.load(file)

fig, axs = plt.subplots(1, 2, figsize=(11, 6))

axs[0].plot(linear_loss["epoch_avg_losses"], label="Linear Schedule Train Loss")
axs[0].plot(cosine_loss["epoch_avg_losses"], label="Cosine Schedule Train Loss")
axs[0].set_xlabel("Epoch", fontsize=12)
axs[0].set_ylabel(r"$\overline{L}_{MSE}$", fontsize=12)
axs[0].legend()

axs[1].plot(linear_loss["test_avg_losses"], label="Linear Schedule Test Loss")
axs[1].plot(cosine_loss["test_avg_losses"], label="Cosine Schedule Test Loss")
axs[1].set_xlabel("Epoch", fontsize=12)
axs[1].set_ylabel(r"$\overline{L}_{MSE}$", fontsize=12)
axs[1].legend()

plt.savefig("../figures/loss_curves.png", dpi=300, bbox_inches="tight")
print("Saved loss_curves.png in figures directory")

# plt.show()

# %%

nearest_loss_path = "../metrics/losses_zoom_NEAREST.json"
bilinear_loss_path = "../metrics/losses_zoom_BILINEAR.json"

with open(nearest_loss_path, "r") as file:
    nearest_loss = json.load(file)
with open(bilinear_loss_path, "r") as file:
    bilinear_loss = json.load(file)

fig, axs = plt.subplots(1, 2, figsize=(10, 6))

axs[0].plot(bilinear_loss["epoch_avg_losses"], label="Bilinear Set Train Loss")
axs[0].plot(nearest_loss["epoch_avg_losses"], label="Nearest Interpolation Train Loss")
axs[0].set_xlabel("Epoch", fontsize=12)
axs[0].set_ylabel(r"$\overline{L}_{MSE}$", fontsize=12)
axs[0].legend()

axs[1].plot(bilinear_loss["test_avg_losses"], label="Bilinear Set Test Loss")
axs[1].plot(nearest_loss["test_avg_losses"], label="Nearest Interpolation Test Loss")
axs[1].set_xlabel("Epoch", fontsize=12)
axs[1].set_ylabel(r"$L_{MSE}$", fontsize=12)
axs[1].legend()

plt.savefig("../figures/loss_curves_zoom.png", dpi=300, bbox_inches="tight")
print("Saved loss_curves_zoom.png in figures directory")

# plt.show()

# %% Plot nearest train and test loss on the same plot
plt.figure(figsize=(8, 6))
plt.plot(nearest_loss["test_avg_losses"], label="Cold Zoom Test Loss", color="black")
plt.plot(nearest_loss["epoch_avg_losses"], label="Cold Zoom Train Loss", color="green")

plt.xlabel("Epoch", fontsize=12)
plt.ylabel(r"$\overline{L}_{MSE}$", fontsize=12)
plt.legend()
plt.savefig("../figures/nearest_loss.png", dpi=300, bbox_inches="tight")
print("Saved nearest_loss.png in figures directory")
# plt.show()

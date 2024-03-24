## @file get_MNIST_pixel_distr
#  @brief This script is used to get the pixel distribution of zoomed images.
#
#  The script facilitates generating the starting point (latent image) for the sampling process.
#  It addresses both NEAREST distribution strategy and BILINEAR set strategy.
#  For NEAREST, the script saves a 4x4 pixel distribution, and for BILINEAR, a 5x5 image set is saved.
#  These are saved as PyTorch tensors. We plot the pixel value distribution of the zoomed images.

# %%
from strat_funcs import single_alternating_zoom_batch
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.transforms import InterpolationMode
import matplotlib.pyplot as plt
import numpy as np


# %%

# Load MNIST dataset
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
mnist_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=tf)

# Create a DataLoader
dataloader = DataLoader(mnist_dataset, batch_size=128, shuffle=False)
# %% Save the BILINEAR zoomed-in set of images
# Zoom level
zoom_level = 23

# Store zoomed images
zoomed_images = []

# Process images
for batch, _ in tqdm(dataloader):
    zoomed_batch = single_alternating_zoom_batch(
        batch, torch.full((batch.size(0),), zoom_level)
    )
    zoomed_images.append(zoomed_batch)

# Concatenate all batches into a single tensor
all_zoomed_images = torch.cat(zoomed_images, dim=0)
# torch.save(all_zoomed_images, f"sample_sets/set_zoom_level_{zoom_level}.pt")

# %% Get the distribution of the 4x4 pixel values for NEAREST interpolation.

# Zoom level
zoom_level = 24

# Store zoomed images
zoomed_images = []

# Process images
for batch, _ in tqdm(dataloader):
    zoomed_batch = single_alternating_zoom_batch(
        batch,
        torch.full((batch.size(0),), zoom_level),
        interpolation=InterpolationMode.NEAREST,
    )
    zoomed_images.append(zoomed_batch)

# Concatenate all batches into a single tensor
all_zoomed_images = torch.cat(zoomed_images, dim=0)
# %%
torch.save(all_zoomed_images, f"sample_sets/set_zoom_level_{zoom_level}_NEAREST.pt")

# %% Get the distribution of pixel values
zoomed_images = torch.load("sample_sets/set_zoom_level_24_NEAREST.pt")
zoomed_images = zoomed_images.numpy()

# Get the distribution of pixel values
pixel_values = zoomed_images.flatten()
plt.hist(pixel_values, bins=256)
plt.title("Pixel Value Distribution of Zoomed Images")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.show()


# Calculate probabilities
bins = np.linspace(-0.5, 0.5, num=257)  # 256 bins from -0.5 to 0.5

# save the bins
torch.save(bins, "sample_sets/bins_zoom_level_24_NEAREST.pt")

counts, _ = np.histogram(pixel_values, bins=256)
probs = counts / counts.sum()

# Convert to PyTorch tensor
probs_tensor = torch.tensor(probs, dtype=torch.float)

# Create a Categorical distribution
distribution = torch.distributions.categorical.Categorical(probs=probs_tensor)

# save the distribution
torch.save(distribution, "sample_sets/distribution_zoom_level_24_NEAREST.pt")

# Sample from the distribution
samples = distribution.sample((10000,))

sampled_pixel_values = bins[samples]
plt.hist(sampled_pixel_values, bins=50, density=True, color="black", alpha=0.5)
# plt.title("Sampled Pixel Value Distribution of Zoomed Images")
plt.xlabel("Central 4x4 Pixel Value")
plt.ylabel("Density")
plt.savefig("../figures/sample_pixel_distr.png", dpi=500, bbox_inches="tight")
plt.show()

# %%
import torch
import random
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.distributions.categorical import Categorical
import numpy as np


# -------------Centre zoom -----------------#


def single_center_crop_resize(img, time_step, interpolation=InterpolationMode.NEAREST):
    _, h, w = img.shape
    target_size = 28 - time_step

    # Calculate center cropping coordinates
    top = (h - target_size) // 2
    left = (w - target_size) // 2

    # Crop and resize
    img_cropped_resized = F.resized_crop(
        img, top, left, target_size, target_size, (h, w), interpolation=interpolation
    )
    return img_cropped_resized


def batch_center_crop_resize(
    images, time_steps, interpolation=InterpolationMode.NEAREST
):
    batch_cropped_resized = []
    for img, t in zip(images, time_steps):
        cropped_resized_img = single_center_crop_resize(img, t.item(), interpolation)
        batch_cropped_resized.append(cropped_resized_img.unsqueeze(0))
    return torch.cat(batch_cropped_resized, dim=0)


def generate_centre_z_T(mean, std, batch_size):
    # Ensure mean and std are float values
    mean = mean.item() if isinstance(mean, torch.Tensor) else mean
    std = std.item() if isinstance(std, torch.Tensor) else std

    # Generate 4 values from the normal distribution for the 4 large blocks
    # Each block is 7x7 pixels
    block_values = torch.normal(mean, std, size=(batch_size, 1, 4, 4))

    # Expand each block to 7x7 to fill the 28x28 image
    expanded_tensor = torch.repeat_interleave(
        torch.repeat_interleave(block_values, 7, dim=2), 7, dim=3
    )
    return expanded_tensor


def extract_center(img, target_size=2):

    _, height, width = img.shape

    # Calculate the top-left corner of the central square
    top = (height - target_size) // 2
    left = (width - target_size) // 2

    # Crop the central part of the image
    center = F.crop(img, top, left, target_size, target_size)

    return center


def single_alternating_zoom(
    img, step, total_steps=28, interpolation=InterpolationMode.NEAREST
):
    """
    Applies a zooming operation to an image by alternately cropping from the top-left and bottom-right corners.
    When training, the step (t) range should be U[1,27] inclusive.

    Parameters:
    - img (torch.Tensor): The input image.
    - step (int): The current zoom step. Ranges from 0 (no zoom) to (total_steps - 1) for maximum zoom.
    - total_steps (int, optional): The total number of steps available for zooming, defaulting to 28. This should be equal to the dimension of the image for a square image (e.g., 28 for a 28x28 image).
    - interpolation (InterpolationMode, optional): The interpolation method used for resizing, defaulting to Nearest.

    Returns:
    - torch.Tensor: The zoomed image.

    The zooming process is designed for a square image (e.g., 28x28 pixels). It alternates between cropping from the top-left and bottom-right corners of the image. Each step progressively zooms in on the image, reducing its effective size. The process allows for a total of 27 zoom steps, leading from an unzoomed state (28x28) to a fully zoomed state (1x1 pixel). The 'total_steps' parameter represents the total number of distinct states, including the original state and the fully zoomed state.

    Examples of zoom steps for a 28x28 image:
    - Step 0: No zoom is applied (28x28 pixels).
    - Step 1: Zooms to 27x27 pixels.
    - Step 2: Zooms to 26x26 pixels.
    - ...
    - Step 27: Zooms to a single pixel (1x1).
    """
    _, h, w = img.shape

    # Determine the amount of reduction for top-left and bottom-right
    top_left_reduction = (step + 1) // 2
    bottom_right_reduction = step // 2

    # Calculate the target size
    target_size = total_steps - top_left_reduction - bottom_right_reduction

    # Make sure we don't go below 1 pixel
    target_size = max(target_size, 1)

    # Calculate the top-left corner for cropping
    top = top_left_reduction
    left = top_left_reduction

    # Calculate the bottom-right corner for cropping
    bottom = h - bottom_right_reduction
    right = w - bottom_right_reduction

    img_cropped_resized = F.resized_crop(
        img, top, left, bottom - top, right - left, (h, w), interpolation=interpolation
    )
    return img_cropped_resized


def single_alternating_zoom_batch(
    images, steps, total_steps=28, interpolation=InterpolationMode.NEAREST
):
    batch_cropped_resized = []
    for img, step in zip(images, steps):
        cropped_resized_img = single_alternating_zoom(
            img, step.item(), total_steps, interpolation
        )
        batch_cropped_resized.append(cropped_resized_img.unsqueeze(0))
    return torch.cat(batch_cropped_resized, dim=0)


def extract_central_pixels_from_loader(dataloader, steps=27):
    central_pixels = []

    for batch, _ in tqdm(dataloader):
        for img in batch:
            # Apply zoom to step 27
            zoomed_img = single_alternating_zoom(img, steps)
            # Take any pixel's value as the image is of uniform color
            central_pixel = zoomed_img.view(-1)[
                0
            ].item()  # Extract the first pixel value
            central_pixels.append(central_pixel)

    return central_pixels


def sample_from_central_pixel_distribution(batch_size):
    # Load the histogram data
    central_pixels = np.load("central_pixels.npy")

    # Calculate probabilities
    bins = np.linspace(-0.5, 0.5, num=257)  # 256 bins from -0.5 to 0.5
    counts, _ = np.histogram(central_pixels, bins=bins)
    probs = counts / counts.sum()

    # Convert to PyTorch tensor
    probs_tensor = torch.tensor(probs, dtype=torch.float)

    # Create a Categorical distribution
    distribution = torch.distributions.categorical.Categorical(probs=probs_tensor)

    # Sample from the distribution
    samples = distribution.sample([batch_size])

    # Convert sample indices to actual pixel values
    sampled_pixel_values = bins[samples]

    # Convert the sampled pixel values to a PyTorch tensor
    # and reshape it to [batch_size, 1, 1, 1] before expanding to [batch_size, 1, 28, 28]
    sampled_images = torch.tensor(sampled_pixel_values, dtype=torch.float)
    sampled_images = sampled_images.view(batch_size, 1, 1, 1).expand(-1, 1, 28, 28)

    return sampled_images


"""

num_samples = 2000  # Number of samples to draw
sampled_pixels = sample_from_central_pixel_distribution(num_samples)

# Plot the distribution of sampled pixel values
plt.hist(sampled_pixels, bins=256)
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.title("Distribution of Sampled Central Pixel in MNIST Training Set (Normalized)")
plt.show()

"""
# %%

# Define the transform and load the dataset
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
train_dataset = MNIST("./data", train=True, download=True, transform=tf)
train_dataloader = DataLoader(
    train_dataset, batch_size=128, shuffle=True, num_workers=0, drop_last=True
)

# Extract central pixels from the dataloader
central_pixels = extract_central_pixels_from_loader(train_dataloader)
# %%
import numpy as np

# Plot the distribution of central pixels
plt.hist(central_pixels, bins=257)
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.title("Distribution of Central Pixel in MNIST Training Set (Normalized)")
plt.show()

counts, bins = np.histogram(central_pixels, bins=256)
probs = counts / len(central_pixels)


np.save("central_pixels.npy", central_pixels)
# %%
# Calculate probabilities
bins = np.linspace(-0.5, 0.5, num=257)  # 256 bins from -0.5 to 0.5
counts, _ = np.histogram(central_pixels, bins=bins)
probs = counts / counts.sum()

# Convert to PyTorch tensor
probs_tensor = torch.tensor(probs, dtype=torch.float)

# Create a Categorical distribution
distribution = Categorical(probs=probs_tensor)

# Sample from the distribution
num_samples = 30000  # Number of samples to draw
samples = distribution.sample([num_samples])

# Convert sample indices to actual pixel values
sampled_pixel_values = bins[samples]

# Plot the distribution of sampled pixel values
plt.hist(sampled_pixel_values, bins=256)
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.title("Distribution of Sampled Central Pixel in MNIST Training Set (Normalized)")
plt.show()

# %%
dataset = MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
img, _ = dataset[2]  # Example image


# Plotting
fig, axes = plt.subplots(1, 6, figsize=(15, 5))  # Change 6 to see more steps
axes[0].imshow(img.squeeze(), cmap="gray")
axes[0].set_title("Original")
axes[0].axis("off")

# Apply function and plot for various steps
steps_to_test = [0, 1, 25, 26, 27]  # Modify as needed to test different steps
for i, step in enumerate(steps_to_test):
    cropped_img = single_alternating_zoom(img, step)
    axes[i + 1].imshow(cropped_img.squeeze(), cmap="gray")
    axes[i + 1].set_title(f"Step {step}")
    axes[i + 1].axis("off")

plt.show()

# %%
"""
# %% Testing single zoom

# Load an example image from MNIST
# dataset = MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
# for i in range(0, 10):
#     img, _ = dataset[i]  # Example image

#     # visualise single_random_crop_resize:

#     zoomed_in = single_center_crop_resize(img, time_step=26)

#     # Visualize the results
#     fig, axes = plt.subplots(1, 2, figsize=(8, 4))
#     axes[0].imshow(img.squeeze(), cmap="gray")
#     axes[0].set_title("Original")
#     axes[0].axis("off")
#     axes[1].imshow(zoomed_in.squeeze(), cmap="gray")
#     axes[1].set_title("Zoomed In")
#     axes[1].axis("off")
#     plt.show()


# # %%

# fig, axes = plt.subplots(4, 4, figsize=(16, 16))

# for i in range(4):
#     for j in range(0, 4, 2):
#         idx = 2 * i + j // 2
#         img, _ = dataset[idx]  # Example image

#         # Apply zoom
#         zoomed_in = single_center_crop_resize(img, time_step=26)

#         # Original image
#         axes[i, j].imshow(img.squeeze(), cmap="gray")
#         axes[i, j].set_title(f"Original {idx}")
#         axes[i, j].axis("off")

#         # Zoomed image
#         axes[i, j + 1].imshow(zoomed_in.squeeze(), cmap="gray")
#         axes[i, j + 1].set_title(f"Zoomed In {idx}")
#         axes[i, j + 1].axis("off")

# plt.show()


# # %%


# # Example usage
# # Assuming 'img' is a 28x28 image from MNIST dataset
# img, _ = dataset[0]  # Replace with actual image from your dataset
# center_region = extract_center(img)
# plt.imshow(center_region.squeeze(), cmap="gray")

# # %%
# # Load MNIST dataset
# transform = transforms.Compose(
#     [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))]
# )
# mnist_dataset = MNIST("./data", train=True, download=True, transform=transform)

# # Initialize lists to store pixel values of central regions
# central_pixels = []

# # Loop over the dataset and extract central 4x4 region from each image
# for img, _ in mnist_dataset:
#     center_region = extract_center(img)
#     central_pixels.append(center_region)

# # Convert list to a single tensor
# central_pixels_tensor = torch.cat(central_pixels, dim=0)

# # Calculate mean and std of central pixels
# mean_central = central_pixels_tensor.mean()
# std_central = central_pixels_tensor.std()

# mean_central, std_central


# # %%

# %%
# # Example usage
mean = torch.tensor([-0.0697])
std = torch.tensor([0.4373])
batch_size = 10  # Number of samples to generate

# Generate images
images = generate_centre_z_T(mean, std, batch_size)
print(images.shape)

# Visualize the images
fig, axs = plt.subplots(1, batch_size, figsize=(20, 2))
for i, img in enumerate(images):
    axs[i].imshow(img.squeeze(), cmap="gray")
    axs[i].axis("off")
plt.show()
"""

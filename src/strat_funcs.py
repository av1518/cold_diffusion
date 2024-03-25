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


def single_alternating_zoom(
    img, step, total_steps=28, interpolation=InterpolationMode.BILINEAR
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
    images, steps, total_steps=28, interpolation=InterpolationMode.BILINEAR
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


def sample_from_set(tensor_set, batch_size):
    sampled_indices = torch.randint(0, tensor_set.size(0), (batch_size,))
    sampled_z_T = tensor_set[sampled_indices]
    return sampled_z_T


def generate_4x4_z_T(batch_size):
    # Load the saved bins and distribution
    bins = torch.load("sample_sets/bins_zoom_level_24_NEAREST.pt")
    distribution = torch.load("sample_sets/distribution_zoom_level_24_NEAREST.pt")

    # Sample from the distribution for each image in the batch
    sampled_pixel_indices = distribution.sample((batch_size, 4, 4))

    # Convert sample indices to actual pixel values and to a PyTorch tensor
    sampled_pixel_values = torch.tensor(bins[sampled_pixel_indices], dtype=torch.float)

    # Initialize an empty tensor for the batch
    batch_tensors = torch.empty((batch_size, 1, 28, 28))

    # Interpolate each tensor to 28x28 and store in the batch
    for i in range(batch_size):
        resized_image = torch.nn.functional.interpolate(
            sampled_pixel_values[i].unsqueeze(0).unsqueeze(0),
            size=(28, 28),
            mode="nearest",
        )
        batch_tensors[i] = resized_image.squeeze(0)

    return batch_tensors


# %%
# Visualise sampling from create_and_interpolate_tensor
# batch_size = 10  # Example batch size
# batch_tensors = generate_4x4_z_T(batch_size)
# print(batch_tensors.shape)  # Should be [batch_size, 1, 28, 28]

# # visualise the images
# fig, axs = plt.subplots(1, batch_size, figsize=(20, 2))
# for i, img in enumerate(batch_tensors):
#     axs[i].imshow(img.squeeze(), cmap="gray")
#     axs[i].axis("off")
# plt.show()

# #print the pixel values
# print(batch_tensors[0].view(-1))  # Print the pixel values of the first image in the batch

# %% Visualize the zoomed images from the sample set
# sample_set = torch.load("sample_sets/set_zoom_level_23.pt")
# sampled_images = sample_from_set(sample_set, 4)

# # Visualize the sampled images
# fig, axs = plt.subplots(1, 4, figsize=(8, 2))
# for i, img in enumerate(sampled_images):
#     axs[i].imshow(img.squeeze(), cmap="gray")
#     axs[i].axis("off")
# plt.show()


# %%

# """

# num_samples = 2000  # Number of samples to draw
# sampled_pixels = sample_from_central_pixel_distribution(num_samples)

# # Plot the distribution of sampled pixel values
# plt.hist(sampled_pixels, bins=256)
# plt.xlabel("Pixel Intensity")
# plt.ylabel("Frequency")
# plt.title("Distribution of Sampled Central Pixel in MNIST Training Set (Normalized)")
# plt.show()

# """
# # %%

# # Define the transform and load the dataset
# tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
# train_dataset = MNIST("./data", train=True, download=True, transform=tf)
# train_dataloader = DataLoader(
#     train_dataset, batch_size=128, shuffle=True, num_workers=0, drop_last=True
# )

# # Extract central pixels from the dataloader
# central_pixels = extract_central_pixels_from_loader(train_dataloader)
# # %%
# import numpy as np

# # Plot the distribution of central pixels
# plt.hist(central_pixels, bins=257)
# plt.xlabel("Pixel Intensity")
# plt.ylabel("Frequency")
# plt.title("Distribution of Central Pixel in MNIST Training Set (Normalized)")
# plt.show()

# counts, bins = np.histogram(central_pixels, bins=256)
# probs = counts / len(central_pixels)


# np.save("central_pixels.npy", central_pixels)
# # %%
# # Calculate probabilities
# bins = np.linspace(-0.5, 0.5, num=257)  # 256 bins from -0.5 to 0.5
# counts, _ = np.histogram(central_pixels, bins=bins)
# probs = counts / counts.sum()

# # Convert to PyTorch tensor
# probs_tensor = torch.tensor(probs, dtype=torch.float)

# # Create a Categorical distribution
# distribution = Categorical(probs=probs_tensor)

# # Sample from the distribution
# num_samples = 30000  # Number of samples to draw
# samples = distribution.sample([num_samples])

# # Convert sample indices to actual pixel values
# sampled_pixel_values = bins[samples]

# # Plot the distribution of sampled pixel values
# plt.hist(sampled_pixel_values, bins=256)
# plt.xlabel("Pixel Intensity")
# plt.ylabel("Frequency")
# plt.title("Distribution of Sampled Central Pixel in MNIST Training Set (Normalized)")
# plt.show()

# # %%
# %%

# # %%
# """


# # # %%

# # fig, axes = plt.subplots(4, 4, figsize=(16, 16))

# # for i in range(4):
# #     for j in range(0, 4, 2):
# #         idx = 2 * i + j // 2
# #         img, _ = dataset[idx]  # Example image

# #         # Apply zoom
# #         zoomed_in = single_center_crop_resize(img, time_step=26)

# #         # Original image
# #         axes[i, j].imshow(img.squeeze(), cmap="gray")
# #         axes[i, j].set_title(f"Original {idx}")
# #         axes[i, j].axis("off")

# #         # Zoomed image
# #         axes[i, j + 1].imshow(zoomed_in.squeeze(), cmap="gray")
# #         axes[i, j + 1].set_title(f"Zoomed In {idx}")
# #         axes[i, j + 1].axis("off")

# # plt.show()


# # # %%


# # # Example usage
# # # Assuming 'img' is a 28x28 image from MNIST dataset
# # img, _ = dataset[0]  # Replace with actual image from your dataset
# # center_region = extract_center(img)
# # plt.imshow(center_region.squeeze(), cmap="gray")

# # # %%
# # # Load MNIST dataset
# # transform = transforms.Compose(
# #     [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))]
# # )
# # mnist_dataset = MNIST("./data", train=True, download=True, transform=transform)

# # # Initialize lists to store pixel values of central regions
# # central_pixels = []

# # # Loop over the dataset and extract central 4x4 region from each image
# # for img, _ in mnist_dataset:
# #     center_region = extract_center(img)
# #     central_pixels.append(center_region)

# # # Convert list to a single tensor
# # central_pixels_tensor = torch.cat(central_pixels, dim=0)

# # # Calculate mean and std of central pixels
# # mean_central = central_pixels_tensor.mean()
# # std_central = central_pixels_tensor.std()

# # mean_central, std_central


# # # %%

# # %%
# # # Example usage
# mean = torch.tensor([-0.0697])
# std = torch.tensor([0.4373])
# batch_size = 10  # Number of samples to generate

# # Generate images
# images = generate_centre_z_T(mean, std, batch_size)
# print(images.shape)

# # Visualize the images
# fig, axs = plt.subplots(1, batch_size, figsize=(20, 2))
# for i, img in enumerate(images):
#     axs[i].imshow(img.squeeze(), cmap="gray")
#     axs[i].axis("off")
# plt.show()
# """

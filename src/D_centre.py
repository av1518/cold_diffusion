# %%
import torch
import random
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader


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


def generate_centre_z_T(mean, std):
    # Ensure mean and std are float values
    mean = mean.item() if isinstance(mean, torch.Tensor) else mean
    std = std.item() if isinstance(std, torch.Tensor) else std

    # Generate 4 values from the normal distribution for the 4 large blocks
    block_values = torch.normal(mean, std, size=(2, 2))

    # Expand each element to a 14x14 block
    expanded_tensor = torch.repeat_interleave(
        torch.repeat_interleave(block_values, 14, dim=0), 14, dim=1
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


# # Example usage
# mean_value = -0.0697
# std_value = 0.4373
# generated_28x28_tensor = generate_centre_z_T(mean_value, std_value)

# # Visualize the generated tensor
# plt.imshow(generated_28x28_tensor, cmap="gray")

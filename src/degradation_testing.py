# %%
import torch
import random
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.transforms import InterpolationMode


# def progressive_zoom(img, steps, zoom_factor):
#     """
#     Progressively zooms in on an image.

#     Parameters:
#     img (PIL.Image or torch.Tensor): The input image.
#     steps (int): Number of zoom steps.
#     zoom_factor (float): Zoom factor per step. Less than 1 for zoom-in.

#     Returns:
#     torch.Tensor: A batch of progressively zoomed images.
#     """
#     if not isinstance(img, torch.Tensor):
#         img = F.to_tensor(img)

#     _, h, w = img.shape
#     zoomed_images = []

#     for step in range(steps):
#         new_h, new_w = int(h * (zoom_factor**step)), int(w * (zoom_factor**step))
#         # Calculate the cropping box
#         top = (h - new_h) // 2
#         left = (w - new_w) // 2
#         img_zoomed = F.resized_crop(img, top, left, new_h, new_w, (h, w))
#         zoomed_images.append(img_zoomed.unsqueeze(0))

#     return torch.cat(zoomed_images, dim=0)


# def progressive_random_zoom(img, steps, zoom_factor):
#     """
#     Progressively zooms in on random parts of an image.

#     Parameters:
#     img (PIL.Image or torch.Tensor): The input image.
#     steps (int): Number of zoom steps.
#     zoom_factor (float): Zoom factor per step. Less than 1 for zoom-in.

#     Returns:
#     torch.Tensor: A batch of progressively zoomed images.
#     """
#     if not isinstance(img, torch.Tensor):
#         img = F.to_tensor(img)

#     _, h, w = img.shape
#     zoomed_images = []

#     for step in range(steps):
#         new_h, new_w = int(h * (zoom_factor**step)), int(w * (zoom_factor**step))

#         # Calculate random top and left coordinates for cropping
#         top = random.randint(0, h - new_h)
#         left = random.randint(0, w - new_w)

#         img_zoomed = F.resized_crop(img, top, left, new_h, new_w, (h, w))
#         zoomed_images.append(img_zoomed.unsqueeze(0))

#     return torch.cat(zoomed_images, dim=0)


def single_zoom(img, zoom_factor, interpolation=InterpolationMode.BILINEAR):
    _, h, w = img.shape

    new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
    new_h = max(1, new_h)
    new_w = max(1, new_w)

    # Print the new dimensions
    # print(f"New dimensions: Height={new_h}, Width={new_w}")

    # Use PyTorch's random functionalities
    top = torch.randint(0, max(h - new_h, 1), (1,)).item()
    left = torch.randint(0, max(w - new_w, 1), (1,)).item()
    # print(f"Top={top}, Left={left}")

    img_zoomed = F.resized_crop(
        img, top, left, new_h, new_w, (h, w), interpolation=interpolation
    )
    return img_zoomed


def progressive_random_zoom(img, steps, zoom_factor):
    """
    Applies the same zoom level multiple times to an image.

    Parameters:
    img (torch.Tensor): The input image.
    steps (int): Number of times the zoom is applied.
    zoom_factor (float): Zoom factor for each step. Less than 1 for zoom-in.

    Returns:
    List[torch.Tensor]: A list of images progressively zoomed.
    """
    zoomed_images = [img]

    for _ in range(steps):
        img = single_zoom(img, zoom_factor)
        zoomed_images.append(img)

    return zoomed_images


def progressive_random_zoom_batch(images, steps, zoom_factor):
    batch_zoomed_images = []
    for img, step in zip(images, steps):
        zoomed_img = progressive_random_zoom(
            img, steps=step.item(), zoom_factor=zoom_factor
        )[-1]
        batch_zoomed_images.append(zoomed_img.unsqueeze(0))
    return torch.cat(batch_zoomed_images, dim=0)


def single_random_crop_resize(img, time_step, interpolation=InterpolationMode.BILINEAR):
    """
    Randomly crops an image to a specified size and resizes it back to the original size.

    Parameters:
    img (torch.Tensor): The input image.
    target_size (int): The target size for both height and width of the crop.
    interpolation (InterpolationMode): The interpolation mode for resizing.

    Returns:
    torch.Tensor: The cropped and resized image.
    """
    _, h, w = img.shape
    # Ensure target size is within the image dimensions and at least 1 pixel
    target_size = 28 - time_step

    # Randomly choose top and left coordinates for cropping
    top = torch.randint(0, h - target_size + 1, (1,)).item()
    left = torch.randint(0, w - target_size + 1, (1,)).item()

    # Crop and resize
    img_cropped_resized = F.resized_crop(
        img, top, left, target_size, target_size, (h, w), interpolation=interpolation
    )
    return img_cropped_resized


# %%

# # Load an example image from MNIST
# dataset = MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
# img, _ = dataset[5]  # Example image

# # visualise single_random_crop_resize:

# zoomed_in = single_random_crop_resize(img, time_step=25)

# # Visualize the results
# fig, axes = plt.subplots(1, 2, figsize=(8, 4))
# axes[0].imshow(img.squeeze(), cmap="gray")
# axes[0].set_title("Original")
# axes[0].axis("off")
# axes[1].imshow(zoomed_in.squeeze(), cmap="gray")
# axes[1].set_title("Zoomed In")
# axes[1].axis("off")
# plt.show()


# %%


# # Load an example image from MNIST
# dataset = MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
# img, _ = dataset[5]  # Example image

# # %% Visualise progressive zoom at centre
# zoomed_images = progressive_zoom(img, steps=10, zoom_factor=0.70)

# # Visualize the results
# fig, axes = plt.subplots(1, 10, figsize=(15, 3))
# for i, ax in enumerate(axes):
#     ax.imshow(zoomed_images[i].squeeze(), cmap="gray")
#     ax.axis("off")
# plt.show()
# # %% Visualise progressive random zoom
# zoomed_images = progressive_random_zoom(img, steps=10, zoom_factor=0.8)

# # Visualize the results
# fig, axes = plt.subplots(1, 10, figsize=(15, 3))
# for i, ax in enumerate(axes):
#     ax.imshow(zoomed_images[i].squeeze(), cmap="gray")
#     ax.axis("off")
# plt.show()


# # %% Progressive random zoom again
# img, _ = dataset[0]  # Example image
# fig, axes = plt.subplots(2, 10, figsize=(20, 4))

# # Plot the original image first
# axes[0, 0].imshow(img.squeeze(), cmap="gray")
# axes[0, 0].axis("off")
# axes[0, 0].set_title("Original")

# # Now plot the progressively zoomed images
# for i in range(1, 20):
#     zoom_factor = 0.99
#     zoomed_img = single_zoom(img, zoom_factor)
#     img = zoomed_img  # Update the image for the next iteration

#     ax = axes[i // 10, i % 10]  # Determine row and column
#     ax.imshow(zoomed_img.squeeze(), cmap="gray")
#     ax.axis("off")
#     ax.set_title(f"Zoom {i}")

# plt.tight_layout()
# plt.show()

# # %%
# img, _ = dataset[4]  # Example image

# example = progressive_random_zoom(img, steps=28, zoom_factor=0.90)[-1]
# plt.imshow(example.squeeze(), cmap="gray")

# # %%
# img, _ = dataset[4]  # Example image

# # Apply the progressive random zoom
# zoomed_images = progressive_random_zoom(img, steps=100, zoom_factor=0.99)

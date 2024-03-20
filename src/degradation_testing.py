# %%
import torch
import random
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader


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
    _, h, w = img.shape
    target_size = 28 - time_step
    top = torch.randint(0, h - target_size + 1, (1,)).item()
    left = torch.randint(0, w - target_size + 1, (1,)).item()
    img_cropped_resized = F.resized_crop(
        img, top, left, target_size, target_size, (h, w), interpolation=interpolation
    )
    return img_cropped_resized


def batch_random_crop_resize(
    images, time_steps, interpolation=InterpolationMode.BILINEAR
):
    batch_cropped_resized = []
    for img, t in zip(images, time_steps):
        cropped_resized_img = single_random_crop_resize(img, t.item(), interpolation)
        batch_cropped_resized.append(cropped_resized_img.unsqueeze(0))
    return torch.cat(batch_cropped_resized, dim=0)


dataset = MNIST(
    root="./data", train=True, download=True, transform=transforms.ToTensor()
)


def process_dataset_at_z_T(dataset, max_steps):
    processed_images = []

    for img, _ in dataset:
        # Process each image at max_step
        processed_img = single_random_crop_resize(img, max_step)
        processed_images.append(processed_img)

    return torch.stack(processed_images)


batch_size = 64  # You can adjust the batch size according to your system's capability
dataset = MNIST(
    root="./data", train=True, download=True, transform=transforms.ToTensor()
)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

# Load the training dataset
train_dataset = MNIST("./data", train=True, download=True, transform=tf)
train_dataloader = DataLoader(
    train_dataset, batch_size=128, shuffle=True, num_workers=0, drop_last=True
)


# Function to process each batch
def process_batch(data_loader, max_step):
    batch_processed = []

    for imgs, _ in data_loader:
        time_steps = torch.full((imgs.size(0),), max_step, dtype=torch.int64)
        processed_batch = batch_random_crop_resize(imgs, time_steps)
        batch_processed.append(processed_batch)

    return torch.cat(batch_processed, dim=0)


# Process the entire dataset
max_step = 27  # Maximum step for cropping
processed_dataset = process_batch(train_dataloader, max_step)

# Calculate mean and std
mean = processed_dataset.mean([0, 2, 3])
std = processed_dataset.std([0, 2, 3])

print(f"Mean: {mean}, Std: {std}")


# %%
def create_uniform_noisy_image(mean, std, image_size=(1, 28, 28), noise_scaling=1):
    """
    Creates a noisy image where all pixels have the same value,
    sampled from a Gaussian distribution. The image is in the format
    expected by the network (mean ~ 0, std ~ 1).

    Parameters:
    mean (float): Mean of the Gaussian distribution.
    std (float): Standard deviation of the Gaussian distribution.
    image_size (tuple): Size of the image (channels, height, width).
    noise_scaling (float): Factor to scale the noise intensity.

    Returns:
    torch.Tensor: Noisy image of size [1, channels, height, width].
    """
    single_value = torch.normal(mean, std * noise_scaling, size=(1,)).item()
    noisy_image = torch.full(image_size, single_value)
    print(single_value)
    return noisy_image.unsqueeze(0)  # Add batch dimension


def rescale_image_for_visualization(image):
    """
    Rescales an image from [-5, 5] range to [0, 1] for visualization.

    Parameters:
    image (torch.Tensor): The input image.

    Returns:
    torch.Tensor: The rescaled image.
    """
    # Rescale from [-5, 5] to [0, 1]
    rescaled_image = (image + 5) / 10
    print(rescaled_image)
    return rescaled_image.clamp(0, 1)


mean_value = mean.item()  # Use the calculated mean from your dataset
std_value = std.item()  # Use the calculated standard deviation
# Create noisy image and rescale for visualization
noisy_image = create_uniform_noisy_image(
    mean_value, std_value, image_size=(28, 28), noise_scaling=5
)
rescaled_noisy_image = rescale_image_for_visualization(noisy_image)

# Visualize the rescaled noisy image
plt.imshow(rescaled_noisy_image.squeeze(), cmap="gray")
plt.show()


# %%

# Load an example image from MNIST
dataset = MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
img, _ = dataset[5]  # Example image

# visualise single_random_crop_resize:

zoomed_in = single_random_crop_resize(img, time_step=27)

# Visualize the results
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(img.squeeze(), cmap="gray")
axes[0].set_title("Original")
axes[0].axis("off")
axes[1].imshow(zoomed_in.squeeze(), cmap="gray")
axes[1].set_title("Zoomed In")
axes[1].axis("off")
plt.show()


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

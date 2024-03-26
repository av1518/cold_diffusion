# %%
import torch
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode
from tqdm import tqdm


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
    """
    @brief Applies alternating zoom to a batch of images with specified zoom steps.

    @param images Tensor: Batch of images ([batch size, channel, height, width]).
    @param steps Tensor: Zoom steps for each image (integers).
    @param total_steps int, optional: Max zoom steps, default 28.
    @param interpolation InterpolationMode, optional: Method for image resizing.

    @return Tensor: Batch of zoomed images.

    Zoom alternates between top-left and bottom-right cropping.
    """
    batch_cropped_resized = []
    for img, step in zip(images, steps):
        cropped_resized_img = single_alternating_zoom(
            img, step.item(), total_steps, interpolation
        )
        batch_cropped_resized.append(cropped_resized_img.unsqueeze(0))
    return torch.cat(batch_cropped_resized, dim=0)


def extract_central_pixels_from_loader(dataloader, steps=27):
    """
    @brief Extracts central pixel from each image in a DataLoader after applying zoom.
           Used in building the central pixel distribution to create seed latents for sampling.

    @param dataloader DataLoader: The DataLoader containing the dataset to process.
    @param steps int, optional: Number of steps to zoom, default is 27.

    @return list: A list containing the central pixel value of each zoomed image.


    """
    central_pixels = []

    for batch, _ in tqdm(dataloader):
        for img in batch:
            zoomed_img = single_alternating_zoom(img, steps)
            central_pixel = zoomed_img.view(-1)[0].item()
            central_pixels.append(central_pixel)

    return central_pixels


def sample_from_set(tensor_set, batch_size):
    """
    @brief Samples a batch of tensors from a given tensor set. Used
           for generating seed latents for sampling using Cold Zoom diffusion
           with Bilinear interpolation.

    @param tensor_set Tensor: The set of tensors to sample from.
    @param batch_size int: The number of samples to draw.

    @return Tensor: A batch of tensors sampled from the provided set.
    """
    sampled_indices = torch.randint(0, tensor_set.size(0), (batch_size,))
    sampled_z_T = tensor_set[sampled_indices]
    return sampled_z_T


def generate_seed_latent(batch_size):
    """
    @brief Generates seed latents for sampling using Cold Zoom diffusion with Nearest-Neighbor
              interpolation. The seed latents are 4x4 pixel images that are resized to 28x28 pixels.

    @param batch_size int: Number of samples to generate in the batch.

    @return Tensor: Batch of images of size [batch_size, 1, 28, 28].

    This function generates seed latents for sampling using Cold Zoom diffusion with Nearest-Neighbor
    interpolation. The seed latents are 4x4 pixel images that are resized to 28x28 pixels. The seed latents
    are sampled from a distribution of central pixel values extracted from the MNIST training set.


    @note Requires 'bins_zoom_level_24_NEAREST.pt' and
    'distribution_zoom_level_24_NEAREST.pt' files in 'sample_sets' directory.
    """

    # Load the saved bins and distribution
    bins = torch.load("sample_sets/bins_zoom_level_24_NEAREST.pt")
    distribution = torch.load("sample_sets/distribution_zoom_level_24_NEAREST.pt")

    # Sample from the distribution for each image in the batch
    sampled_pixel_indices = distribution.sample((batch_size, 4, 4))
    sampled_pixel_values = torch.tensor(bins[sampled_pixel_indices], dtype=torch.float)

    # Create empty tensor for the batch
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

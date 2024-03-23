# %%
from nn_Gaussian import DDPM, CNN
from nn_zoom_4x4_distr import DDPM_custom
from torchvision.datasets import MNIST
from torchvision import transforms
import torch
import torch.nn as nn
from torchmetrics.image.fid import FrechetInceptionDistance
import json
import os

# %%


def calculate_FID(n_compare, real_dataset, model_to_sample, model_device):
    """
    Calculate FID using standard inception network on GPU.

    Args:
    n_compare (int): Number of samples to compare.
    real_dataset (torchvision.datasets): Dataset containing real images.
    model_to_sample (torch.nn.Module): Model to generate fake samples.
    model_device (torch.device): Device to perform calculations (GPU).

    Returns:
    float: Computed FID score.
    """

    # Clone the real dataset and convert to 3 channel images, keep on GPU
    real = real_dataset.data.clone().unsqueeze(1).repeat(1, 3, 1, 1)
    real = real.to(model_device, dtype=torch.float32)
    # Normalize to [0, 1] and then convert to 8-bit unsigned integers
    real = (real / 2 + 0.5) * 255
    real = real.to(torch.uint8)
    n_real = real.shape[0]
    assert n_compare <= n_real

    metric = FrechetInceptionDistance()
    metric.to(model_device)  # Move metric computation to GPU

    # Update metric with real samples
    metric.update(real[:n_compare], real=True)

    model_to_sample.eval()
    with torch.no_grad():
        # Generate samples and process for FID computation
        samples = model_to_sample.sample(n_compare, model_device)
        samples = samples.repeat(1, 3, 1, 1)  # Ensure samples have 3 channels
        # Normalize and convert to 8-bit unsigned integers
        samples = (samples / 2 + 0.5) * 255
        samples = samples.to(torch.uint8)

        # Update metric with generated samples
        metric.update(samples, real=False)

    # Compute FID score
    fid_score = metric.compute()
    return fid_score.item()  # Convert to Python float if needed


# %%
n_hidden = (16, 32, 32, 16)
device = "cuda" if torch.cuda.is_available() else "cpu"
n_T = 23

# Initialize DDPM model
gt = CNN(in_channels=1, expected_shape=(28, 28), n_hidden=n_hidden, act=nn.GELU)
gt.to(device)
ddpm = DDPM_custom(gt=gt, n_T=n_T, criterion=nn.MSELoss())
ddpm.to(device)

# Load MNIST dataset
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0))])
dataset = MNIST("./data", train=True, download=True, transform=tf)


# Directory for saving metrics
metrics_dir = os.path.join("..", "metrics")
os.makedirs(metrics_dir, exist_ok=True)

fid_metrics = {}


# FID Calculation and Saving Scores for each epoch
for epoch in range(0, 110, 10):
    model_path = f"../saved_models/ddpm_alt_BI_5x5_set_{epoch}.pth"

    # Load the model
    ddpm.load_state_dict(torch.load(model_path, map_location=device))
    ddpm.eval()

    # Calculate FID score
    fid_score = calculate_FID(100, dataset, ddpm, device)
    print(f"Epoch {epoch}: FID Score = {fid_score}")

    # Save the FID score for the epoch
    fid_metrics[f"Epoch {epoch}"] = fid_score

# Save the FID metrics to a file in the metrics folder
fid_metrics_filepath = os.path.join(metrics_dir, "fid_zoom_5x5_set.json")
with open(fid_metrics_filepath, "w") as f:
    json.dump(fid_metrics, f, indent=4)

print(f"FID scores saved to {fid_metrics_filepath}")

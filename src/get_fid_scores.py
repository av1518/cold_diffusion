## @file get_fid_scores.py
#  @brief This script is used for evaluating the performance of our models on the MNIST dataset.
#
#  It loads the MNIST dataset,initializes the models, and calculates the Fréchet Inception Distance (FID) score for the models.
#  The FID scores are calculated for specified epochs and saved into a JSON file in the 'metrics' directory.
# %%
from nn_ddpm import CNN
from nn_zoom_nearest import DDPM_zoom_4x4_distr
from torchvision.datasets import MNIST
from torchvision import transforms
import torch
import torch.nn as nn
import json
import os
from utils import calculate_FID

# %%
n_hidden = (16, 32, 32, 16)
device = "cuda" if torch.cuda.is_available() else "cpu"
n_T = 24

# Initialize DDPM model
gt = CNN(in_channels=1, expected_shape=(28, 28), n_hidden=n_hidden, act=nn.GELU)
gt.to(device)
ddpm = DDPM_zoom_4x4_distr(gt=gt, n_T=n_T, criterion=nn.MSELoss())
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
    model_path = f"../saved_models/alt_ddpm_NEAREST_4x4_distr_{epoch}.pth"

    # Load the model
    ddpm.load_state_dict(torch.load(model_path, map_location=device))
    ddpm.eval()

    # Calculate FID score
    fid_score = calculate_FID(100, dataset, ddpm, device)
    print(f"Epoch {epoch}: FID Score = {fid_score}")

    # Save the FID score for the epoch
    fid_metrics[f"Epoch {epoch}"] = fid_score
# %%
# Save the FID metrics to a file in the metrics folder
fid_metrics_filepath = os.path.join(metrics_dir, "fid_zoom_4x4_distr.json")
with open(fid_metrics_filepath, "w") as f:
    json.dump(fid_metrics, f, indent=4)

print(f"FID scores saved to {fid_metrics_filepath}")

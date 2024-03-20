# %%
import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid

from models import DDPM, CNN

tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0))])
# Load the training dataset
train_dataset = MNIST("./data", train=True, download=True, transform=tf)
train_dataloader = DataLoader(
    train_dataset, batch_size=128, shuffle=True, num_workers=4, drop_last=True
)

# Load the test dataset
test_dataset = MNIST("./data", train=False, download=True, transform=tf)
test_dataloader = DataLoader(
    test_dataset, batch_size=128, shuffle=False, num_workers=4, drop_last=True
)

gt = CNN(in_channels=1, expected_shape=(28, 28), n_hidden=(16, 32, 32, 16), act=nn.GELU)
# For testing: (16, 32, 32, 16)
# For more capacity (for example): (64, 128, 256, 128, 64)
ddpm = DDPM(gt=gt, betas=(1e-4, 0.02), n_T=1000)
optim = torch.optim.Adam(ddpm.parameters(), lr=2e-4)
# %%
accelerator = Accelerator()

# We wrap our model, optimizer, and dataloaders with `accelerator.prepare`,
# which lets HuggingFace's Accelerate handle the device placement and gradient accumulation.
ddpm, optim, dataloader = accelerator.prepare(ddpm, optim, train_dataloader)
print("Device:", accelerator.device)

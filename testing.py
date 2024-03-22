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
from D_centre import (
    single_alternating_zoom_batch,
    sample_from_central_pixel_distribution,
)
from models import CNN
import torch.nn as nn

tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

train_dataset = MNIST("./data", train=True, download=True, transform=tf)
train_dataloader = DataLoader(
    train_dataset, batch_size=3, shuffle=True, num_workers=0, drop_last=True
)

for x, _ in train_dataloader:
    print(x.shape)
    print(x[0].shape)
    print(x[0].squeeze(0).shape)
    print(x.size(0))  # batch size
    break
# %%
batch_size = x.size(0)
# plot the images in the batch
fig, ax = plt.subplots(1, 3)
for i in range(batch_size):
    ax[i].imshow(x[i].squeeze(0), cmap="gray")

plt.show()


# %%

t_batch = torch.randint(1, 27 + 1, (batch_size,), device=x.device)
print("t_batch:", t_batch)
# %%
t_batch = torch.tensor([5, 21, 27], device=x.device)
z_t = single_alternating_zoom_batch(x, t_batch)
print(z_t.shape)

# plot the images in the batch
fig, ax = plt.subplots(1, 3)
for i in range(batch_size):
    ax[i].imshow(z_t[i].squeeze(0), cmap="gray")

plt.show()

# %%
t_batch.float() / 27

# %%

z_T = sample_from_central_pixel_distribution(3, 3)

print(z_T.shape)
# print the first value of each tensor in the batch
print(z_T[0, 0, 0][0])
print(z_T[1, 0, 0][0])
print(z_T[2, 0, 0][0])

# plot the images in the batch
fig, ax = plt.subplots(1, 3)
for i in range(batch_size):

    ax[i].imshow(z_T[i].squeeze(0), cmap="gray", vmin=-0.5, vmax=0.5)
    # remove the axis
    ax[i].axis("off")
    # add the title
    ax[i].set_title(f"z_T[{i}]")
plt.show()

# %%
for s in range(27, 0, -1):
    print(s)

# %%
s = 27
scaled_time = (s / 27) * torch.ones(3, device="cuda")
print(scaled_time)
gt = CNN(in_channels=1, expected_shape=(28, 28), n_hidden=(16, 32, 32, 16), act=nn.GELU)
gt.to("cuda")
# %%
z_T = sample_from_central_pixel_distribution(3, 3)
z_T = z_T.to("cuda")

for s in range(27, 0, -1):
    scaled_time = (s / 27) * torch.ones(3, device="cuda")
    scaled_time = scaled_time.to("cuda")
    x_hat = gt(z_T, scaled_time)
    # print(x_hat.shape)
    steps = torch.full((3,), s, device="cuda")
    print(steps)
    D_0_s = single_alternating_zoom_batch(x_hat, torch.full((3,), s, device="cuda"))

    D_0_s_1 = single_alternating_zoom_batch(
        x_hat, torch.full((3,), s - 1, device="cuda")
    )
    z_T -= D_0_s - D_0_s_1

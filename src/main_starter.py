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
ddpm, optim, train_dataloader = accelerator.prepare(ddpm, optim, train_dataloader)
print("Device:", accelerator.device)
# %% make sure this works
for x, _ in train_dataloader:
    print(x.shape)
    print(x.device)
    break

with torch.no_grad():
    print(ddpm(x))
    print("Passed initial test")
# %% Training

n_epoch = 100
losses = []

for i in range(n_epoch):
    ddpm.train()

    pbar = tqdm(train_dataloader)  # Wrap our loop with a visual progress bar
    for x, _ in pbar:
        optim.zero_grad()
        loss = ddpm(x)
        accelerator.backward(loss)
        # ^Technically should be `accelerator.backward(loss)` but not necessary for local training
        losses.append(loss.item())
        avg_loss = np.average(
            losses[max(len(losses) - 100, 0) :]
        )  # calculates the current average loss
        pbar.set_description(f"loss: {avg_loss:.3g}")
        optim.step()

    ddpm.eval()
    with torch.no_grad():
        xh = ddpm.sample(n_sample=16, size=(1, 28, 28), device=accelerator.device)
        # Can get device explicitly with `accelerator.device`
        # ^ make 16 samples, The size of each sample to generate (excluding the batch dimension).
        # This should match the expected input size of the model.
        grid = make_grid(xh, nrow=4)

        # Save samples to `./contents` directory
        save_image(grid, f"./contents/ddpm_sample_{i:04d}.png")

        # save model
        torch.save(ddpm.state_dict(), f"./ddpm_mnist.pth")

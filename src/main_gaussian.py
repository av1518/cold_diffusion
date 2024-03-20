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


# %%
def setup_dataloaders():
    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))]
    )

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

    return train_dataloader, test_dataloader


def main():
    train_dataloader, test_dataloader = setup_dataloaders()

    # Test dataloader
    for x, _ in train_dataloader:
        print(x.shape, x.device)
        break

    # Rest of your script that uses the dataloaders


if __name__ == "__main__":
    main()

# Model and Optimizer definitions (outside the main function)
gt = CNN(in_channels=1, expected_shape=(28, 28), n_hidden=(16, 32, 32, 16), act=nn.GELU)
ddpm = DDPM(gt=gt, betas=(1e-4, 0.02), n_T=1000)
optim = torch.optim.Adam(ddpm.parameters(), lr=2e-4)

accelerator = Accelerator()
ddpm, optim = accelerator.prepare(ddpm, optim)

# Testing the model (can also be inside main if it involves data loading)
with torch.no_grad():
    # Dummy input to test the model
    dummy_input = torch.randn(128, 1, 28, 28).to(accelerator.device)
    ddpm(dummy_input)
    print("Passed initial test")
    # %% Training

    n_epoch = 2
    moving_avg_loss = []
    epoch_avg_losses = []  # List to store average loss per epoch

    for i in range(n_epoch):
        ddpm.train()
        pbar = tqdm(train_dataloader)  # Wrap our loop with a visual progress bar
        epoch_losses = []  # List to store losses for each batch in the current epoch

        for x, _ in pbar:
            optim.zero_grad()
            loss = ddpm(x)
            accelerator.backward(loss)
            # ^Technically should be `accelerator.backward(loss)` but not necessary for local training
            moving_avg_loss.append(loss.item())
            avg_loss = np.average(
                moving_avg_loss[max(len(moving_avg_loss) - 100, 0) :]
            )  # calculates the current average loss
            pbar.set_description(f"loss: {avg_loss:.3g}")
            optim.step()

            loss_item = loss.item()
            epoch_losses.apppend(loss_item)

        epoch_avg_loss = np.mean(epoch_losses)
        epoch_avg_losses.append(epoch_avg_loss)

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

if __name__ == "__main__":
    main()

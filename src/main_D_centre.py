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
import matplotlib.pyplot as plt


import json
from datetime import datetime
from models import CNN
from nn_D_centre import DDPM_custom

# %%
# def main():

tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

# Load the training dataset
train_dataset = MNIST("./data", train=True, download=True, transform=tf)
train_dataloader = DataLoader(
    train_dataset, batch_size=128, shuffle=True, num_workers=0, drop_last=True
)

# Load the test dataset
test_dataset = MNIST("./data", train=False, download=True, transform=tf)
test_dataloader = DataLoader(
    test_dataset, batch_size=128, shuffle=False, num_workers=0, drop_last=True
)

gt = CNN(in_channels=1, expected_shape=(28, 28), n_hidden=(16, 32, 32, 16), act=nn.GELU)
# For testing: (16, 32, 32, 16)
# For more capacity (for example): (64, 128, 256, 128, 64)
ddpm = DDPM_custom(gt=gt, n_T=26, criterion=nn.MSELoss())
optim = torch.optim.Adam(ddpm.parameters(), lr=2e-4)

accelerator = Accelerator()

# We wrap our model, optimizer, and dataloaders with `accelerator.prepare`,
# which lets HuggingFace's Accelerate handle the device placement and gradient accumulation.
ddpm, optim, train_dataloader = accelerator.prepare(ddpm, optim, train_dataloader)
print("Device:", accelerator.device)
# make sure this works
for x, _ in train_dataloader:
    print(x.shape, x.device)
    break

with torch.no_grad():
    ddpm(x)
    print(ddpm(x))
    print("Passed initial test")

# %% Training

n_epoch = 50
moving_avg_loss = []
epoch_avg_losses = []  # List to store average loss per epoch
test_avg_losses = []

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
        pbar.set_description(f"100 Moving average Loss: {avg_loss:.3g}")
        optim.step()

        loss_item = loss.item()
        epoch_losses.append(loss_item)

    epoch_avg_loss = np.mean(epoch_losses)
    epoch_avg_losses.append(epoch_avg_loss)

    ddpm.eval()
    test_losses = []
    with torch.no_grad():
        for x, _ in test_dataloader:
            x = x.to(accelerator.device)
            loss = ddpm(x)
            test_losses.append(loss.item())

        test_avg_loss = np.mean(test_losses)
        test_avg_losses.append(test_avg_loss)

        xh = ddpm.sample(n_samples=16, device=accelerator.device)
        # Can get device explicitly with `accelerator.device`
        # ^ make 16 samples, The size of each sample to generate (excluding the batch dimension).
        # This should match the expected input size of the model.
        grid = make_grid(xh, nrow=4)

        # Save samples to `./contents` directory
        save_image(grid, f"./contents_custom/ddpm_sample_{i:04d}.png")

        # save model
        # torch.save(ddpm.state_dict(), f"./ddpm_mnist.pth")
# %%
# After training, plot and save the loss curve
plt.plot(range(1, n_epoch + 1), epoch_avg_losses, label="Average Loss per Epoch")
# plt.plot(range(1, n_epoch + 1), test_avg_losses, label="Test Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.title("DDPM Training Loss Curve")
plt.legend()
# plt.savefig("./contents/ddpm_loss_curve.png")
plt.show()

# %%


def save_metrics_to_json(filename, data):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


# Save average loss per epoch to a JSON file with timestamp
current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
metrics_filename = f"./contents/ddpm_metrics_{current_date}.json"
metrics_data = {
    "epoch_avg_losses": epoch_avg_losses,
    "test_avg_losses": test_avg_losses,
}
save_metrics_to_json(metrics_filename, metrics_data)


# if __name__ == '__main__':
#     main()

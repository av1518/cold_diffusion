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
from nn_ddpm import DDPM, CNN


# %%
# parameters
n_hidden = (16, 32, 32, 16)
betas = (1e-4, 0.02)
noise_scheduler = "linear"
n_epoch = 100

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

gt = CNN(in_channels=1, expected_shape=(28, 28), n_hidden=n_hidden, act=nn.GELU)
ddpm = DDPM(gt=gt, betas=betas, n_T=1000, noise_scheduler=noise_scheduler)
optim = torch.optim.Adam(ddpm.parameters(), lr=2e-4)

accelerator = Accelerator()
ddpm, optim, train_dataloader = accelerator.prepare(ddpm, optim, train_dataloader)
print("Device:", accelerator.device)

# make sure this works
for x, _ in train_dataloader:
    print(x.shape, x.device)
    break

with torch.no_grad():
    ddpm(x)
    print("Passed initial test")

# %% Training
moving_avg_loss = []
epoch_avg_losses = []
test_avg_losses = []

for i in range(n_epoch):
    ddpm.train()
    pbar = tqdm(train_dataloader)  # Wrap our loop with a visual progress bar
    epoch_losses = []  # List to store losses for each batch in the current epoch

    for x, _ in pbar:
        optim.zero_grad()
        loss = ddpm(x)
        accelerator.backward(loss)
        moving_avg_loss.append(loss.item())
        avg_loss = np.average(
            moving_avg_loss[max(len(moving_avg_loss) - 100, 0) :]
        )  # calculates the current average loss
        pbar.set_description(f"100 Moving average Loss: {avg_loss:.3g}, Epoch: {i}")
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

        xh = ddpm.sample(n_sample=16, size=(1, 28, 28), device=accelerator.device)
        grid = make_grid(xh, nrow=4)

        save_image(grid, f"./contents/ddpm_gaussian_linear_sample_{i:04d}.png")

        if i % 5 == 0:
            torch.save(
                ddpm.state_dict(), f"../saved_models/ddpm_gaussian_linear_{i}.pth"
            )

torch.save(ddpm.state_dict(), f"../saved_models/ddpm_gaussian_linear_{n_epoch}.pth")

# %% Plot the loss curves
# After training, plot and save the loss curve
plt.plot(range(1, n_epoch + 1), epoch_avg_losses, label="Average Loss per Epoch")
plt.plot(range(1, n_epoch + 1), test_avg_losses, label="Test Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.title("DDPM Training Loss Curve")
plt.legend()
plt.savefig("../figures/ddpm_gaussian_linear_loss_curve.png")
plt.show()

# %% Save metrics to JSON file


def save_metrics_to_json(filename, data):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


metrics_filename = f"../metrics/ddpm_gaussian_linear.json"
metrics_data = {
    "epoch_avg_losses": epoch_avg_losses,
    "test_avg_losses": test_avg_losses,
}
save_metrics_to_json(metrics_filename, metrics_data)

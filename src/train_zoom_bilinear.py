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
import wandb


import json
from nn_ddpm import CNN
from nn_zoom_bilinear import DDPM_zoom_5x5_set

# %%
# Parameters
learning_rate = 2e-4
batch_size = 128
n_T = 23


tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

# Load the training dataset
train_dataset = MNIST("./data", train=True, download=True, transform=tf)
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
)

# Load the test dataset
test_dataset = MNIST("./data", train=False, download=True, transform=tf)
test_dataloader = DataLoader(
    test_dataset, batch_size=128, shuffle=False, num_workers=0, drop_last=True
)

gt = CNN(in_channels=1, expected_shape=(28, 28), n_hidden=(16, 32, 32, 16), act=nn.GELU)
# For testing: (16, 32, 32, 16)
# For more capacity (for example): (64, 128, 256, 128, 64)
ddpm = DDPM_zoom_5x5_set(gt=gt, n_T=n_T, criterion=nn.MSELoss())
optim = torch.optim.Adam(ddpm.parameters(), lr=learning_rate)

accelerator = Accelerator()

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


n_epoch = 100
moving_avg_loss = []
epoch_avg_losses = []  # List to store average loss per epoch
test_avg_losses = []

wandb.init(project="Custom-Diffusion", entity="av662")

wandb.config = {
    "learning_rate": learning_rate,
    "epochs": n_epoch,
    "batch_size": batch_size,
}


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
        wandb.log({"avg_train_loss": loss_item})

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
        wandb.log({"avg_test_loss": test_avg_loss})

        xh = ddpm.sample(n_samples=16, device=accelerator.device)
        grid = make_grid(xh, nrow=4)

        # Save samples to `./contents` directory
        save_image(grid, f"./contents_custom/alt_ddpm_BI_5x5_set_{i:04d}.png")

        # save model every 10 epochs
        if i % 10 == 0:
            torch.save(
                ddpm.state_dict(), f"../saved_models/ddpm_alt_BI_5x5_set_{i}.pth"
            )

torch.save(ddpm.state_dict(), f"../saved_models/ddpm_alt_BI_5x5_set_{n_epoch}.pth")

wandb.finish()
# %% Plot the loss curves
# After training, plot and save the loss curve
plt.plot(range(1, n_epoch + 1), epoch_avg_losses, label="Average Loss per Epoch")
plt.plot(range(1, n_epoch + 1), test_avg_losses, label="Test Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.title("DDPM Training Loss Curve")
plt.legend()
plt.savefig("./contents_custom/ddpm_loss_curve_alt_BI.png")
plt.show()


# %% Save metrics to JSON file
def save_metrics_to_json(filename, data):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


metrics_filename = f"../saved_models/metrics_ddpm_alt_BI_5x5_set.json"


metrics_data = {
    "epoch_avg_losses": epoch_avg_losses,
    "test_avg_losses": test_avg_losses,
}
save_metrics_to_json(metrics_filename, metrics_data)

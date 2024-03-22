# %%
from strat_funcs import single_alternating_zoom_batch
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random
from tqdm import tqdm


# Load MNIST dataset
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
mnist_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=tf)

# Create a DataLoader
dataloader = DataLoader(mnist_dataset, batch_size=128, shuffle=False)

# Zoom level
zoom_level = 23

# Store zoomed images
zoomed_images = []

# Process images
for batch, _ in tqdm(dataloader):
    zoomed_batch = single_alternating_zoom_batch(
        batch, torch.full((batch.size(0),), zoom_level)
    )
    zoomed_images.append(zoomed_batch)

# Concatenate all batches into a single tensor
all_zoomed_images = torch.cat(zoomed_images, dim=0)
torch.save(all_zoomed_images, f"sample_sets/set_zoom_level_{zoom_level}.pt")


# %%
# Function to pick a random image from the zoomed dataset
def pick_random_image(zoomed_dataset):
    random_idx = random.randint(0, zoomed_dataset.size(0) - 1)
    return zoomed_dataset[random_idx]


# Pick a random zoomed image
random_zoomed_image = pick_random_image(all_zoomed_images)

print(random_zoomed_image.shape)
# Show the random image (using matplotlib or any other library)
import matplotlib.pyplot as plt

plt.imshow(random_zoomed_image.squeeze(), cmap="gray")
plt.title("Random Zoomed Image")
plt.axis("off")
plt.show()

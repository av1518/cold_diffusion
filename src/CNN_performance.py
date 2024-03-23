# %%
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from nn_D_centre_alt_1_pix import DDPM_custom
from nn_Gaussian import CNN
from strat_funcs import single_alternating_zoom
from torch import nn

# Parameters

epoch = 90


model_path = f"../saved_models/ddpm_alt_BI_{epoch}.pth"
n_hidden = (16, 32, 32, 16)

epochs_to_load = 50  # Epoch of the model you want to load
num_images = 5  # Number of images to process
zoom_levels = [10, 15, 20, 25, 27]  # Different levels of zoom

gt = CNN(in_channels=1, expected_shape=(28, 28), n_hidden=(16, 32, 32, 16), act=nn.GELU)
model = DDPM_custom(gt=gt, n_T=27, criterion=nn.MSELoss())
model.load_state_dict(torch.load(model_path))

# Load the MNIST dataset
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
test_dataset = datasets.MNIST("../data", train=False, download=True, transform=tf)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=num_images, shuffle=True
)


# %%
def display_images(images, titles):
    plt.figure(figsize=(len(images) * 2, 4))
    for i, (img, title) in enumerate(zip(images, titles)):
        if img.requires_grad:
            img = img.detach().cpu().numpy().squeeze()
        else:
            img = img.squeeze()
        plt.subplot(1, len(images), i + 1)
        plt.imshow(img, cmap="gray")
        plt.title(title)
        plt.axis("off")
    plt.show()


# Process images
images, _ = next(iter(test_loader))
num_images = len(images)

original_titles = ["Original" for _ in range(num_images)]
zoom_titles = [f"Zoom Level: {zoom_level}" for zoom_level in zoom_levels[:num_images]]
recon_titles = [f"Reconstructed" for _ in range(num_images)]

originals, zoomed_in, reconstructions = [], [], []

for i, img in enumerate(images):
    originals.append(img.squeeze())
    zoom_level = zoom_levels[i % len(zoom_levels)]  # Get the corresponding zoom level
    zoomed_img = single_alternating_zoom(img, zoom_level)
    zoomed_in.append(zoomed_img.squeeze())
    recon_img = model.gt(
        zoomed_img.unsqueeze(0), torch.tensor([zoom_level / 27])
    ).squeeze()
    reconstructions.append(recon_img)

# Visualize originals, zoomed in, and reconstructions
display_images(originals, original_titles)
display_images(zoomed_in, zoom_titles)
display_images(reconstructions, recon_titles)

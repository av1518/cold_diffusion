# %%
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from nn_zoom_4x4_distr import DDPM_zoom_4x4_distr
from nn_Gaussian import CNN
from strat_funcs import single_alternating_zoom
from torch import nn
from torchvision.transforms import InterpolationMode
import torchvision


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


def display_all_images(
    originals,
    zoomed_in,
    reconstructions,
    original_titles,
    zoom_titles,
    recon_titles,
    num=1,
):
    num_images = len(originals)
    fig, axes = plt.subplots(
        3, num_images, figsize=(2 * num_images, 6)
    )  # 3 rows for each image type

    for i in range(num_images):
        # Original image
        ax = axes[0, i]
        img = originals[i].detach() if originals[i].requires_grad else originals[i]
        ax.imshow(img, cmap="gray")
        ax.set_title(original_titles[i])
        ax.axis("off")

        # Zoomed image
        ax = axes[1, i]
        img = zoomed_in[i].detach() if zoomed_in[i].requires_grad else zoomed_in[i]
        ax.imshow(img, cmap="gray")
        ax.set_title(zoom_titles[i])
        ax.axis("off")

        # Reconstructed image
        ax = axes[2, i]
        img = (
            reconstructions[i].detach()
            if reconstructions[i].requires_grad
            else reconstructions[i]
        )
        ax.imshow(img, cmap="gray")
        ax.set_title(recon_titles[i])
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"../figures/CNN_performance_{num}", dpi=500, bbox_inches="tight")
    plt.show()


# %%
# Parameters

epoch = 100


model_path = f"../saved_models/alt_ddpm_NEAREST_4x4_distr_{epoch}.pth"
n_hidden = (16, 32, 32, 16)


num_images = 5  # Number of images to process
zoom_levels = [5, 10, 15, 20, 24]  # Different levels of zoom

gt_distr = CNN(
    in_channels=1, expected_shape=(28, 28), n_hidden=(16, 32, 32, 16), act=nn.GELU
)
model = DDPM_zoom_4x4_distr(gt=gt_distr, n_T=24, criterion=nn.MSELoss())
model.load_state_dict(torch.load(model_path))

# Load the MNIST dataset
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
test_dataset = datasets.MNIST("../data", train=False, download=True, transform=tf)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=num_images, shuffle=True
)


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
    zoomed_img = single_alternating_zoom(
        img, zoom_level, interpolation=InterpolationMode.NEAREST
    )
    zoomed_in.append(zoomed_img.squeeze())
    recon_img = model.gt(
        zoomed_img.unsqueeze(0), torch.tensor([zoom_level / 27])
    ).squeeze()
    reconstructions.append(recon_img)

# Visualize originals, zoomed in, and reconstructions
display_images(originals, original_titles)
display_images(zoomed_in, zoom_titles)
display_images(reconstructions, recon_titles)

# %%
display_all_images(
    originals,
    zoomed_in,
    reconstructions,
    original_titles,
    zoom_titles,
    recon_titles,
    num=4,
)
# %%
# Parameters

epoch = 100

model_path = f"../saved_models/ddpm_alt_BI_{epoch}.pth"
n_hidden = (16, 32, 32, 16)

num_images = 5  # Number of images to process
zoom_levels = [5, 10, 15, 20, 24]  # Different levels of zoom

gt_distr = CNN(
    in_channels=1, expected_shape=(28, 28), n_hidden=(16, 32, 32, 16), act=nn.GELU
)
model = DDPM_zoom_4x4_distr(gt=gt_distr, n_T=24, criterion=nn.MSELoss())
model.load_state_dict(torch.load(model_path))


original_titles = ["Original" for _ in range(num_images)]
zoom_titles = [f"Zoom Level: {zoom_level}" for zoom_level in zoom_levels[:num_images]]
recon_titles = [f"Reconstructed" for _ in range(num_images)]

originals, zoomed_in, reconstructions = [], [], []

for i, img in enumerate(images):
    originals.append(img.squeeze())
    zoom_level = zoom_levels[i % len(zoom_levels)]  # Get the corresponding zoom level
    zoomed_img = single_alternating_zoom(
        img, zoom_level, interpolation=InterpolationMode.BILINEAR
    )
    zoomed_in.append(zoomed_img.squeeze())
    recon_img = model.gt(
        zoomed_img.unsqueeze(0), torch.tensor([zoom_level / 27])
    ).squeeze()
    reconstructions.append(recon_img)

# Visualize originals, zoomed in, and reconstructions
display_images(originals, original_titles)
display_images(zoomed_in, zoom_titles)
display_images(reconstructions, recon_titles)

# %%
display_all_images(
    originals,
    zoomed_in,
    reconstructions,
    original_titles,
    zoom_titles,
    recon_titles,
    num=5,
)

# %%

epoch = 100


model_path = f"../saved_models/alt_ddpm_NEAREST_4x4_distr_{epoch}.pth"
n_hidden = (16, 32, 32, 16)


gt_distr = CNN(
    in_channels=1, expected_shape=(28, 28), n_hidden=(16, 32, 32, 16), act=nn.GELU
)
model = DDPM_zoom_4x4_distr(gt=gt_distr, n_T=24, criterion=nn.MSELoss())
model.load_state_dict(torch.load(model_path))
with torch.no_grad():
    steps = model.sample(n_samples=5, device="cpu", keep_steps=True)


# Assuming 'tensor_list' is your list of tensors
tensor_list = steps

# Select every 5th tensor starting from the first
selected_tensors = [tensor_list[i] for i in range(0, 25, 3)]

# Concatenate these tensors along the batch dimension
concatenated_tensors = torch.cat(selected_tensors, dim=0)

# Add 0.5 to remove the normalisation
normalized_tensors = concatenated_tensors + 0.5

# Use make_grid to create a grid of these images
grid = torchvision.utils.make_grid(normalized_tensors, nrow=5)

# Convert the grid to a numpy array for displaying
grid_np = grid.numpy().transpose((1, 2, 0))

# Display the grid
plt.figure(figsize=(10, 10))
plt.imshow(grid_np, cmap="gray")
plt.axis("off")
plt.savefig(
    f"../figures/sample_progression_4x4_distr_every_3_NEAR_1.png",
    dpi=500,
    bbox_inches="tight",
)
plt.show()
# %%

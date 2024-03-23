# %%
import torch
from torch import nn
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import torch
from nn_Gaussian import DDPM, CNN
from nn_zoom_4x4_distr import DDPM_zoom_4x4_distr
from nn_zoom_5x5_set import DDPM_zoom_5x5_set

n_hidden = (16, 32, 32, 16)
betas = (1e-4, 0.02)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DDPM_linear_path = f"../saved_models/ddpm_gaussian_100.pth"
DDPM_cosine_path = f"../saved_models/ddpm_gaussian_cosine_100.pth"
DDPM_zoom_4x4_distr_path = f"../saved_models/alt_ddpm_NEAREST_4x4_distr_100.pth"
DDPM_zoom_5x5_set_path = f"../saved_models/ddpm_alt_BI_5x5_set_100.pth"

gt_linear = CNN(in_channels=1, expected_shape=(28, 28), n_hidden=n_hidden, act=nn.GELU)
gt_cosine = CNN(in_channels=1, expected_shape=(28, 28), n_hidden=n_hidden, act=nn.GELU)
gt_4x4_distr = CNN(
    in_channels=1, expected_shape=(28, 28), n_hidden=n_hidden, act=nn.GELU
)
gt_5x5_set = CNN(in_channels=1, expected_shape=(28, 28), n_hidden=n_hidden, act=nn.GELU)

ddpm_linear = DDPM(gt=gt_linear, betas=betas, n_T=1000, noise_scheduler="linear")
ddpm_linear.load_state_dict(torch.load(DDPM_linear_path))
ddpm_linear.to(device)
ddpm_linear.eval()

ddpm_cosine = DDPM(gt=gt_cosine, betas=betas, n_T=1000, noise_scheduler="cosine")
ddpm_cosine.load_state_dict(torch.load(DDPM_cosine_path))
ddpm_cosine.to(device)
ddpm_cosine.eval()

ddpm_zoom_4x4_distr = DDPM_zoom_4x4_distr(
    gt=gt_4x4_distr, n_T=24, criterion=nn.MSELoss()
)
ddpm_zoom_4x4_distr.load_state_dict(torch.load(DDPM_zoom_4x4_distr_path))
ddpm_zoom_4x4_distr.to(device)
ddpm_zoom_4x4_distr.eval()

ddpm_zoom_5x5_set = DDPM_zoom_5x5_set(gt=gt_5x5_set, n_T=23, criterion=nn.MSELoss())
ddpm_zoom_5x5_set.load_state_dict(torch.load(DDPM_zoom_5x5_set_path))
ddpm_zoom_5x5_set.to(device)
ddpm_zoom_5x5_set.eval()

# %%
zoom_4x4_good_samples = []
zoom_4x4_bad_samples = []
# %%
sample = []
with torch.no_grad():
    sampled_image = ddpm_zoom_5x5_set.sample(1, device=device)
    sample.append(sampled_image)


# Concatenate all samples into a single tensor
all_samples = torch.cat(sample, dim=0)

# Create a grid of images
grid = make_grid(all_samples, nrow=4)

# Convert grid to a format suitable for showing with matplotlib
# Permute the axes from (C, H, W) to (H, W, C) and normalize
grid = grid.permute(1, 2, 0).cpu().numpy()
grid = (grid - grid.min()) / (grid.max() - grid.min())

# Display the grid of images
plt.imshow(grid, vmin=-0.5, vmax=0.5, cmap="gray")
plt.axis("off")
plt.show()

print(len(zoom_4x4_good_samples))
print(len(zoom_4x4_bad_samples))
# %%
zoom_4x4_good_samples.append(sample)
# %%
zoom_4x4_bad_samples.append(sample)

# %%
# Save good and bad samples as tensors
torch.save(zoom_4x4_good_samples, "zoom_5x5_good_samples.pt")
torch.save(zoom_4x4_bad_samples, "zoom_5x5_samples.pt")


# %%


# %%

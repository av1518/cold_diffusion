## @file plot_schedules_progression
#  @brief Plots linear and cosine noise schedules, plot steps in the forward diffusion
#  process for both DDPM strategies and zoom-in degradation strategies.
#
#
#  This script generates linear and cosine noise schedules, then applies them to an
#  MNIST image for visualization. It compares original and noisy images, and showcases
#  the noise impact at different timesteps for both schedules. We also plot the noise
#  levels for all time steps. Finally, we demonstrate the zoom-in degradation strategy
#  forward diffusion process for both nearest and bilinear interpolation methods.


# %%
import matplotlib.pyplot as plt
from utils import ddpm_schedules, ddpm_cosine_schedules, add_gaussian_noise
from torchvision.datasets import MNIST
from torchvision import transforms
from strat_funcs import single_alternating_zoom, InterpolationMode

# %%
betas = (1e-4, 0.02)
T = 1000

# Generate the linear and cosine schedules
linear_schedules = ddpm_schedules(betas[0], betas[1], T)
cosine_schedules = ddpm_cosine_schedules(T, s=0.008)

# Plotting
fig, ax = plt.subplots(1, 2, figsize=(10, 6))

# Linear Beta_t and Cosine Beta_t plot
ax[0].plot(linear_schedules["beta_t"], label="Linear schedule")
ax[0].plot(cosine_schedules["beta_t"], label="Cosine schedule")
ax[0].set_xlabel("Diffusion Step t", fontsize=11)
ax[0].set_ylabel(r"$\beta_{t}$", fontsize=14)
ax[0].legend()

# Linear Alpha_t and Cosine Alpha_t plot
ax[1].plot(linear_schedules["alpha_t"], label="Linear schedule")
ax[1].plot(cosine_schedules["alpha_t"], label="Cosine schedule")
ax[1].set_xlabel("Diffusion Step t", fontsize=11)
ax[1].set_ylabel(r"$\alpha_t$", fontsize=14)
ax[1].legend()

plt.savefig("../figures/noise_schedules.png", dpi=500, bbox_inches="tight")
print("Saved noise_schedules.png in figures directory")
# plt.show()
# %% plot only alpha_t
fig, ax = plt.subplots(1, 1, figsize=(6, 5))
ax.plot(linear_schedules["alpha_t"], label="Linear schedule")
ax.plot(cosine_schedules["alpha_t"], label="Cosine schedule")
ax.set_xlabel("Diffusion Step t", fontsize=11)
ax.set_ylabel(r"$\alpha_t$", fontsize=14)
ax.legend()
plt.savefig("../figures/noise_schedule.png", dpi=300, bbox_inches="tight")
print("Saved noise_schedule.png in figures directory")
# plt.show()

# %% Add Gaussian noise to an MNIST image
# Load an MNIST image
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

# Load the training dataset
dataset = MNIST("./data", train=True, download=True, transform=tf)
x, _ = dataset[0]
x = x.unsqueeze(0)

# Add Gaussian noise using the linear schedule
x_noisy_linear = add_gaussian_noise(x, t=10, noise_schedule="linear")

# Display the original and noisy images
fig, ax = plt.subplots(1, 2, figsize=(8, 4))

ax[0].imshow(x.squeeze(0).squeeze(0), cmap="gray")
ax[0].set_title("Original Image")
ax[0].axis("off")

ax[1].imshow(x_noisy_linear.squeeze(0).squeeze(0), cmap="gray")
ax[1].set_title("Noisy Image (Linear Schedule)")
ax[1].axis("off")

# plt.show()

# %%
fig, axes = plt.subplots(2, 6, figsize=(10, 4))

# set (timesteps)
noise_levels = [0, 50, 100, 150, 200, 250]

# plot the images with linear and cosine noise
for i, level in enumerate(noise_levels):
    # Linear noise
    x_noisy_linear = add_gaussian_noise(x, t=level, noise_schedule="linear")
    axes[0, i].imshow(x_noisy_linear.squeeze(0).squeeze(0), cmap="gray")
    axes[0, i].set_title(f"Linear t={level}")
    axes[0, i].axis("off")

    # Cosine noise
    x_noisy_cosine = add_gaussian_noise(x, t=level, noise_schedule="cosine")
    axes[1, i].imshow(x_noisy_cosine.squeeze(0).squeeze(0), cmap="gray")
    axes[1, i].set_title(f"Cosine t={level}")
    axes[1, i].axis("off")

plt.tight_layout()
plt.savefig("../figures/noise_progression.png", dpi=300, bbox_inches="tight")
print("Saved noise_progression.png in figures directory")
# plt.show()
# %% Zoom-in Degradation strategy example
img, _ = dataset[0]  # Example image

fig, axes = plt.subplots(1, 6, figsize=(15, 5))
axes[0].imshow(img.squeeze(), cmap="gray")
axes[0].set_title("Original")
axes[0].axis("off")


# Nearest interpolation zoom
steps_to_test = [5, 10, 15, 20, 24]
for i, step in enumerate(steps_to_test):
    cropped_img = single_alternating_zoom(
        img, step, interpolation=InterpolationMode.NEAREST
    )
    axes[i + 1].imshow(cropped_img.squeeze(), cmap="gray")
    axes[i + 1].set_title(f"Step {step}")
    axes[i + 1].axis("off")

plt.savefig("../figures/zoom_NEAREST_steps.png", dpi=300, bbox_inches="tight")
print("Saved zoom_NEAREST_steps.png in figures directory")
# plt.show()

fig, axes = plt.subplots(1, 6, figsize=(15, 5))
axes[0].imshow(img.squeeze(), cmap="gray")
axes[0].set_title("Original")
axes[0].axis("off")

# Bilinear interpolation zoom
steps_to_test = [5, 10, 15, 20, 24]
for i, step in enumerate(steps_to_test):
    cropped_img = single_alternating_zoom(
        img, step, interpolation=InterpolationMode.BILINEAR
    )
    axes[i + 1].imshow(cropped_img.squeeze(), cmap="gray")
    axes[i + 1].set_title(f"Step {step}")
    axes[i + 1].axis("off")

plt.savefig("../figures/zoom_BILINEAR_steps.png", dpi=300, bbox_inches="tight")
print("Saved zoom_BILINEAR_steps.png in figures directory")
# plt.show()

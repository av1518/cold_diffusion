# %%
import matplotlib.pyplot as plt
from utils import ddpm_schedules, ddpm_cosine_schedules


# Example parameters
betas = (1e-4, 0.02)
T = 1000

# Generate the linear and cosine schedules
linear_schedules = ddpm_schedules(betas[0], betas[1], T)
cosine_schedules = ddpm_cosine_schedules(T, s=0.008)

# Plotting
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Linear Beta_t and Cosine Beta_t plot
ax[0].plot(linear_schedules["beta_t"], label="Linear beta_t")
ax[0].plot(cosine_schedules["beta_t"], label="Cosine beta_t")
ax[0].set_title("Beta Schedule Comparison")
ax[0].set_xlabel("Timestep")
ax[0].set_ylabel("Beta Value")
ax[0].legend()

# Linear Alpha_t and Cosine Alpha_t plot
ax[1].plot(linear_schedules["alpha_t"], label="Linear alpha_t")
ax[1].plot(cosine_schedules["alpha_t"], label="Cosine alpha_t")
ax[1].set_title("Alpha Schedule Comparison")
ax[1].set_xlabel("Timestep")
ax[1].set_ylabel("Alpha Value")
ax[1].legend()

plt.show()
# %% plot only alpha_t
fig, ax = plt.subplots(1, 1, figsize=(6, 5))
ax.plot(linear_schedules["alpha_t"], label="Linear schedule")
ax.plot(cosine_schedules["alpha_t"], label="Cosine schedule")
ax.set_xlabel("Diffusion Step t", fontsize=11)
ax.set_ylabel(r"$\alpha_t$", fontsize=14)
ax.legend()
plt.savefig("../figures/noise_schedule.png", dpi=300, bbox_inches="tight")
plt.show()

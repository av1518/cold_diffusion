import torch
from typing import Dict
from torchmetrics.image.fid import FrechetInceptionDistance


def ddpm_schedules(beta1: float, beta2: float, T: int) -> Dict[str, torch.Tensor]:
    """Returns pre-computed schedules for DDPM sampling with a linear noise schedule."""
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    alpha_t = torch.exp(
        torch.cumsum(torch.log(1 - beta_t), dim=0)
    )  # Cumprod in log-space (better precision)

    return {"beta_t": beta_t, "alpha_t": alpha_t}


def f_t(t, T, s):
    """
    Helper function for cosine schedule generation. Returns the value of the cosine
    function at timestep `t` given the total number of steps `T` and the hyperparameter `s`.

    @param t: The current timestep.
    @param T: The total number of diffusion steps.
    @param s: A hyperparameter for schedule adjustment.

    @return: The value of the cosine function at timestep `t`.
    """
    # Ensure t is a Tensor, even if it's a single value
    t = torch.tensor(t, dtype=torch.float32) if not torch.is_tensor(t) else t
    return torch.cos((t / T + s) / (1 + s) * torch.pi / 2) ** 2


def ddpm_cosine_schedules(T: int, s=0.008) -> dict:
    """
    Generates noise schedule for Denoising Diffusion Probabilistic Models (DDPM)
    using a cosine schedule, as proposed by Nichol and Dhariwal (2021).

    @param T: The number of diffusion steps. Determines the length of the schedule.
    @param s: A hyperparameter for schedule adjustment (default: 0.008).

    @return: A dictionary containing:
             - 'beta_t': A torch.Tensor of beta values for each diffusion step.
             - 'alpha_t': A torch.Tensor of alpha values, representing the cumulative
                          product of `(1 - beta_t)` values.

    References:
        Nichol, A., & Dhariwal, P. (2021). Improved Denoising Diffusion Probabilistic Models.
    """
    t = torch.arange(0, T + 1, dtype=torch.float32)

    alpha_t = f_t(t, T, s) / f_t(0, T, s)
    print(alpha_t)
    alpha_t_1 = f_t(t - 1, T, s) / f_t(0, T, s)
    print(alpha_t_1)

    beta_t = torch.clamp(1 - alpha_t / alpha_t_1, 0, 0.02)

    alpha_t = torch.exp(torch.cumsum(torch.log(1 - beta_t), dim=0))
    return {"beta_t": beta_t, "alpha_t": alpha_t}


def add_gaussian_noise(x: torch.Tensor, t: int, noise_schedule="linear"):
    """
    Adds Gaussian noise to the input tensor `x` at diffusion step `t`.
    This is for demonstration purposes only and should not be used in training.

    @param x: The input tensor.
    @param t: The diffusion step.

    @return: The input tensor with added Gaussian noise.
    """
    epsilon = torch.randn_like(x)  # noise tensor in the same shape as x

    if noise_schedule == "linear":
        noise_schedule = ddpm_schedules(1e-4, 0.02, 1000)

    if noise_schedule == "cosine":
        noise_schedule = ddpm_cosine_schedules(1000, s=0.008)

    alpha_t = noise_schedule["alpha_t"][t]
    z_t = torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * epsilon

    return z_t


def calculate_FID(n_compare, real_dataset, model_to_sample, model_device):
    """
    Calculate FID using standard inception network for saved models.

    Args:
    n_compare (int): Number of samples to compare.
    real_dataset (torchvision.datasets): Dataset containing real images.
    model_to_sample (torch.nn.Module): Model to generate fake samples.
    model_device (torch.device): Device to perform calculations (GPU).

    Returns:
    float: Computed FID score.
    """

    # Clone the real dataset and convert to 3 channel images, keep on GPU
    real = real_dataset.data.clone().unsqueeze(1).repeat(1, 3, 1, 1)
    real = real.to(model_device, dtype=torch.float32)
    # Normalize to [0, 1] and then convert to 8-bit unsigned integers
    real = (real / 2 + 0.5) * 255
    real = real.to(torch.uint8)
    n_real = real.shape[0]
    assert n_compare <= n_real

    metric = FrechetInceptionDistance()
    metric.to(model_device)  # Move metric computation to GPU

    # Update metric with real samples
    metric.update(real[:n_compare], real=True)

    model_to_sample.eval()
    with torch.no_grad():
        # Generate samples and process for FID computation
        samples = model_to_sample.sample(n_compare, model_device)
        samples = samples.repeat(1, 3, 1, 1)  # Ensure samples have 3 channels
        # Normalize and convert to 8-bit unsigned integers
        samples = (samples / 2 + 0.5) * 255
        samples = samples.to(torch.uint8)

        # Update metric with generated samples
        metric.update(samples, real=False)

    # Compute FID score
    fid_score = metric.compute()
    return fid_score.item()  # Convert to Python float if needed

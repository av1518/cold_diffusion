import torch
from typing import Dict


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

    beta_t = 1 - alpha_t / alpha_t_1

    alpha_t = torch.exp(torch.cumsum(torch.log(1 - beta_t), dim=0))
    return {"beta_t": beta_t, "alpha_t": alpha_t}


def add_gaussian_noise(x: torch.Tensor, t: int, noise_schedule="linear"):
    """
    Adds Gaussian noise to the input tensor `x` at diffusion step `t`.

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
    print(alpha_t)
    # alpha_t = alpha_t[t, None, None, None]  # Add singleton dimensions for broadcasting

    z_t = torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * epsilon

    return z_t

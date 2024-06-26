from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from strat_funcs import single_alternating_zoom_batch, generate_seed_latent
from torchvision.transforms import InterpolationMode

sample_set = torch.load("sample_sets/set_zoom_level_23.pt")


class DDPM_zoom_4x4_distr(nn.Module):
    def __init__(
        self,
        gt,
        n_T: int,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super().__init__()

        self.gt = gt
        self.n_T = n_T  # Number of time steps to use in training # this should be 27
        self.criterion = criterion  # loss metric

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is of shape [batch_size, channels, height, width]
        batch_size = x.size(0)
        # print("Batch size:", batch_size)

        # Generate a random step for each image in the batch
        t_batch = torch.randint(1, self.n_T + 1, (batch_size,), device=x.device)
        # print("t_batch:", t_batch)
        # Apply progressive random zoom to the entire batch
        z_t = single_alternating_zoom_batch(
            x, t_batch, interpolation=InterpolationMode.NEAREST
        )
        # print("z_t shape:", z_t.shape)

        # Since criterion is MSELoss, it expects input and target of the same dimensions
        # Loss is the MSE loss between the input x and the cnn output
        # cnn takes in the degraded image and the time step
        return self.criterion(x, self.gt(z_t, t_batch.float() / self.n_T))

    def sample(self, n_samples: int, device, keep_steps=False) -> torch.Tensor:
        steps = []

        z_T = generate_seed_latent(n_samples)
        # shape of z_T is [num_samples, 1, 28, 28]
        z_T = z_T.to(device)

        steps.append(z_T)

        for s in range(self.n_T, 0, -1):
            # go from s = 27 to s = 1
            scaled_time = (s / self.n_T) * torch.ones(n_samples, device=device)
            # ^ shape of scaled_time is [num_samples]
            x_hat = self.gt(z_T, scaled_time)
            z_T = single_alternating_zoom_batch(
                x_hat,
                torch.full((n_samples,), s - 1, device=device),
                interpolation=InterpolationMode.NEAREST,
            )
            steps.append(z_T)

        if keep_steps:
            return steps
        else:
            return z_T

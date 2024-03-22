from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from D_centre import (
    single_alternating_zoom_batch,
    sample_from_central_pixel_distribution,
)


class DDPM_custom(nn.Module):
    def __init__(
        self,
        gt,
        n_T: int,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super().__init__()

        self.gt = gt
        self.n_T = n_T  # Number of time steps
        self.criterion = criterion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is of shape [batch_size, channels, height, width]
        batch_size = x.size(0)

        # Generate a random step for each image in the batch
        t_batch = torch.randint(1, self.n_T, (batch_size,), device=x.device)

        # Apply progressive random zoom to the entire batch
        z_t = single_alternating_zoom_batch(x, t_batch)

        # Since criterion is MSELoss, it expects input and target of the same dimensions
        return self.criterion(z_t, self.gt(z_t, t_batch.float() / self.n_T))

    def sample(self, n_samples: int, device) -> torch.Tensor:
        z_t = sample_from_central_pixel_distribution(256, n_samples)
        z_t = z_t.to(device)

        for s in range(self.n_T, 0, -1):
            scaled_time = (s / self.n_T) * torch.ones(n_samples, device=device)
            x_hat = self.gt(z_t, scaled_time)
            D_0_s = single_alternating_zoom_batch(
                x_hat, torch.full((n_samples,), s, device=device)
            )
            if s > 1:
                D_0_s_1 = single_alternating_zoom_batch(
                    x_hat, torch.full((n_samples,), s - 1, device=device)
                )
                z_t -= D_0_s - D_0_s_1

        return z_t
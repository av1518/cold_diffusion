from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from D_centre import batch_center_crop_resize, generate_centre_z_T


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
        z_t = batch_center_crop_resize(x, t_batch)

        # Since criterion is MSELoss, it expects input and target of the same dimensions
        return self.criterion(z_t, self.gt(z_t, t_batch.float() / self.n_T))

    def sample(self, n_samples: int, device) -> torch.Tensor:
        mean = torch.tensor([-0.0697])
        std = torch.tensor([0.4373])
        z_t = generate_centre_z_T(mean=mean, std=std)
        z_t = (
            z_t.repeat(n_samples, 1, 1).unsqueeze(1).to(device)
        )  # Shape: [n_samples, 1, 28, 28]

        for s in range(self.n_T, 0, -1):
            scaled_time = (s / self.n_T) * torch.ones(n_samples, device=device)
            x_hat = self.gt(z_t, scaled_time.unsqueeze(1))
            z_t -= batch_center_crop_resize(
                x_hat, torch.tensor([s] * n_samples).to(device)
            )
            if s > 1:
                z_t += batch_center_crop_resize(
                    x_hat, torch.tensor([s - 1] * n_samples).to(device)
                )

        return z_t

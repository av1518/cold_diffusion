from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from degradation_testing import progressive_random_zoom_batch, single_random_crop_resize


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
        t = torch.randint(1, self.n_T + 1, (batch_size,), device=x.device)

        # Apply progressive random zoom to the entire batch
        z_t = progressive_random_zoom_batch(x, t, zoom_factor=0.9)

        # Since criterion is MSELoss, it expects input and target of the same dimensions
        return self.criterion(z_t, self.gt(z_t, t.float() / self.n_T))

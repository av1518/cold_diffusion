from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from utils import ddpm_schedules, ddpm_cosine_schedules


class CNNBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        expected_shape,
        act=nn.GELU,
        kernel_size=7,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.LayerNorm(expected_shape),
            act(),
        )

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(
        self,
        in_channels,
        expected_shape=(28, 28),  # MNIST images are 28x28
        n_hidden=(64, 128, 64),  # number channels in each hidden layer
        kernel_size=7,  # size of the convolutional kernel in each layer
        last_kernel_size=3,  # size of the kernel in the last layer
        time_embeddings=16,  # dimensionality of the time embedding vector
        act=nn.GELU,
    ) -> None:
        super().__init__()
        last = in_channels

        self.blocks = nn.ModuleList()
        for hidden in n_hidden:
            self.blocks.append(
                CNNBlock(
                    last,
                    hidden,
                    expected_shape=expected_shape,
                    kernel_size=kernel_size,
                    act=act,
                )
            )
            last = hidden

        # The final layer, we use a regular Conv2d to get the
        # correct scale and shape (and avoid applying the activation)
        self.blocks.append(
            nn.Conv2d(
                last,
                in_channels,
                last_kernel_size,
                padding=last_kernel_size // 2,
            )
        )

        ## This part is literally just to put the single scalar "t" into the CNN
        ## in a nice, high-dimensional way:
        self.time_embed = nn.Sequential(
            nn.Linear(time_embeddings * 2, 128),
            act(),
            nn.Linear(128, 128),
            act(),
            nn.Linear(128, 128),
            act(),
            nn.Linear(128, n_hidden[0]),
        )
        frequencies = torch.tensor(
            [0] + [2 * np.pi * 1.5**i for i in range(time_embeddings - 1)]
        )
        self.register_buffer("frequencies", frequencies)

    def time_encoding(self, t: torch.Tensor) -> torch.Tensor:
        phases = torch.concat(
            (
                torch.sin(t[:, None] * self.frequencies[None, :]),
                torch.cos(t[:, None] * self.frequencies[None, :]) - 1,
            ),
            dim=1,
        )

        return self.time_embed(phases)[:, :, None, None]

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Shapes of input:
        #    x: (batch, chan, height, width)
        #    t: (batch,)

        embed = self.blocks[0](x)
        # ^ (batch, n_hidden[0], height, width)

        # Add information about time along the diffusion process
        #  (Providing this information by superimposing in latent space)
        embed += self.time_encoding(t)
        #         ^ (batch, n_hidden[0], 1, 1) - thus, broadcasting
        #           to the entire spatial domain

        for block in self.blocks[1:]:
            embed = block(embed)

        return embed


class DDPM(nn.Module):
    def __init__(
        self,
        gt,
        betas: Tuple[float, float],
        n_T: int,
        criterion: nn.Module = nn.MSELoss(),
        noise_scheduler="linear",
    ) -> None:
        super().__init__()

        self.gt = gt

        if noise_scheduler == "linear":
            noise_schedule = ddpm_schedules(betas[0], betas[1], n_T)

        elif noise_scheduler == "cosine":
            noise_schedule = ddpm_cosine_schedules(n_T)

        else:
            raise ValueError(
                f"Invalid noise scheduler: {noise_scheduler}. Please use 'linear' or 'cosine'."
            )

        # `register_buffer` will track these tensors for device placement, but
        # not store them as model parameters. This is useful for constants.
        self.register_buffer("beta_t", noise_schedule["beta_t"])
        self.beta_t  # Exists! Set by register_buffer
        self.register_buffer("alpha_t", noise_schedule["alpha_t"])
        self.alpha_t

        self.n_T = n_T  # Number of time steps
        self.criterion = criterion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Algorithm 18.1 in Prince
        Executes one step of the forward process in the Denoising Diffusion Probabilistic Model.

        This method applies the diffusion process to the input tensor `x` and calculates
        the loss based on the model's ability to predict the noise added at a randomly
        chosen time step. It effectively simulates adding noise to the image and challenges
        the model to predict this noise.

        @param x: Input tensor representing the batch of images.
        @type x: torch.Tensor

        @return: Loss value representing how well the model predicts the noise.
        @rtype: torch.Tensor

        Detailed Algorithm Steps:
        1. Randomly select a time step `t` for each image in the batch.
        2. Generate a noise tensor `epsilon` with the same shape as `x`, following
        a standard normal distribution N(0, 1).
        3. Retrieve and reshape the corresponding alpha values `alpha_t` for the selected
        time steps.
        4. Compute `z_t`, a noisy version of the input images, as a weighted sum of the
        original images and the noise tensor. The weights are determined by `alpha_t`
        and its complement.
        5. Use the CNN model (`self.gt`) to predict the noise `epsilon` from the noisy
        images `z_t`.
        6. Return the loss calculated by comparing the predicted noise with the actual
        noise added, using the specified criterion (`self.criterion`).

        Note:
        The method corresponds to Algorithm 18.1 in the 'Prince' reference, which outlines
        the forward process in diffusion models.
        """

        t = torch.randint(
            1, self.n_T, (x.shape[0],), device=x.device
        )  # Randomly sample a time step, range is [1, n_T)
        epsilon = torch.randn_like(
            x
        )  # eps ~ N(0, 1) #noise tensor with same shape as x
        alpha_t = self.alpha_t[t, None, None, None]  # Get right shape for broadcasting
        # ?[batch_size, channels, height, width]

        z_t = torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * epsilon
        # This is the z_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this z_t. Loss is what we return.

        return self.criterion(
            epsilon, self.gt(z_t, t / self.n_T)
        )  # Criterion is MSELoss

    def sample(self, n_sample: int, size, device) -> torch.Tensor:
        """
        Algorithm 18.2 in Prince
        Generates samples using the reverse diffusion process described in Algorithm 18.2 of the 'Prince' reference.

        This method starts with random noise and applies the reverse steps of the diffusion
        process to generate images. It iteratively refines the noise by predicting and
        subtracting the added noise at each previous timestep, effectively reversing the
        diffusion process to produce an image.

        @param n_sample: Number of samples to generate.
        @type n_sample: int

        @param size: The size of each sample to generate (excluding the batch dimension).
                    This should match the expected input size of the model.
        @type size: tuple

        @param device: The device (CPU or GPU) where the tensors should be allocated.
        @type device: torch.device

        @return: A tensor containing generated samples.
        @rtype: torch.Tensor

        Detailed Algorithm Steps:
        1. Initialize a tensor `z_t` following a standard normal distribution N(0, 1). This is
        the last latent variable.
        2. Iterate backwards through the diffusion timesteps:
        a. For each timestep `i`, calculate `alpha_t` and `beta_t` from the precomputed noise schedule.
        b. Adjust `z_t` by predicting the noise at timestep `i` using the model (`self.gt`) and
            subtract this from `z_t`. This step partially reverses the noise addition of the forward process.
        c. Normalize `z_t` for the current timestep's noise level.
        d. If not at the last timestep, add a new sample of Gaussian noise to `z_t`.
        3. After completing all timesteps, return the final `z_t` tensor, which now contains generated samples.
        """

        _one = torch.ones(n_sample, device=device)
        z_t = torch.randn(n_sample, *size, device=device)

        for i in range(self.n_T, 0, -1):
            print(i)
            alpha_t = self.alpha_t[i]
            beta_t = self.beta_t[i]
            # First line of loop:
            z_t -= (beta_t / torch.sqrt(1 - alpha_t)) * self.gt(
                z_t, (i / self.n_T) * _one
            )
            z_t /= torch.sqrt(1 - beta_t)

            if i > 1:
                # Last line of loop:
                z_t += torch.sqrt(beta_t) * torch.randn_like(z_t)
            # (We don't add noise at the final step - i.e., the last line of the algorithm)
        return z_t

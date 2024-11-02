# M2: Application of Machine Learning Coursework
# Cold Zoom Diffusion


### Description
We develop Cold Zoom Diffusion: a deterministic way of training a diffusion model to produce MNIST digits by continually zooming into the image and extrapolating as our degradation strategy. The Cold Zoom Diffusion model performs comparably to the linear DDPM model, despite its deterministic nature and smaller reconstructor network. However, it shows a bias towards certain digits due to the lack of randomness in the reconstruction phase.


Full report available in `report/`.

### Sample Progression

Below is an example of the sample progression during the diffusion process:
<p align="center">
    <img src="figures/sample_progression_4x4_distr_every_3_NEAR.png" alt="Sample Progression" width="400"/>
</p>


We include scripts for Bilinear Cold Zoom Diffusion, but these results are not discussed in the report. These are here for completeness.



### Installation
Dependencies required to run the project are listed in the `environment.yml` file. To install the necessary Conda environment from these dependencies, run:
```bash
conda env create -f environment.yml
```

Once the environment is created, activate it using: 

```bash
conda activate m2
```

### Script Contents
All scripts have a description of their usage at the top. They are broken into the following sections:



#### Training Scripts
| Script                    | Usage                                                                    |
|---------------------------|--------------------------------------------------------------------------|
| `train_ddpm.py`           | Train Denoising Diffusion Probabilistic Models (DDPM)      |
| `train_zoom_bilinear.py`  | Train Cold Zoom Diffusion using bilinear interpolation |
| `train_zoom_nearest.py`   | Train Cold Zoom Diffusion nearest neighbor interpolation |

#### Evaluation & Figures Scripts
| Script                    | Usage                                                                    |
|---------------------------|--------------------------------------------------------------------------|
| `get_fid_scores.py`       | Calculate FID scores for model evaluation                      |
| `get_good_bad_samples.py` | Separate good and bad samples from model outputs               |
| `get_minist_pixel_dist.py`| Analyze pixel distribution in the MNIST dataset                 |
| `plot_fid_samples_loss.py` | Load and plot FID scores, good and bad generated samples, and loss curves   |
| `plot_reconstruction_sampling.py` | Plot reconstructor (reverse process) network predictions, and various steps during reverse process (sampling)          |
| `plot_schedules_progression.py` | Plots linear and cosine noise schedules, plot steps in the forward diffusion process for both DDPM strategies and zoom-in degradation strategies.        |

#### Utility & Network Scripts
| Script                    | Usage                                                                    |
|---------------------------|--------------------------------------------------------------------------|
| `nn_ddpm.py`              | Neural network  for DDPM                                           |
| `nn_zoom_bilinear.py`     | Neural network  for Cold Zoom Diffusion with bilinear interpolation    |
| `nn_zoom_nearest.py`      | Neural network  for Cold Zoom Diffusion with nearest neighbor interpolation |
| `strat_funcs.py`          | Functions used in Cold Diffusion degradation strategy & evaluation                     |
| `utils.py`                | Utility functions for general purposes in the project                    |



- All the code was ran on a CPU: Ryzen 9, 16gb of RAM, GPU: NVIDIA RTX 3060
- The FID score plots in `get_fid_scores.py` need every model parameters for 20 epochs. These were ommited from the git directory. These can be saved by running the corresponding training files first.


## Dockerfile Instructions
The user can build and run the solver in a Docker container using the `Dockerfile` provided in the repository. From the root directory, build the image with:

```bash
$ docker build -t m2 .
```

This generates a Docker image called `m2`. To deploy and run the package in the container with a local input file, run the following command:

```bash
$ docker run --rm -ti m2
```

This setup uses a virtual Ubuntu environment with Miniconda, installs the necessary packages and activates the environment. From here, simply follow the Usage section to use the solver.




### Contributing

Contributions are welcome. Please open an issue to discuss significant changes and update tests as appropriate.

### License
This project is open-sourced under the [MIT](https://choosealicense.com/licenses/mit/) License.

# M2: Application of Machine Learning Coursework
# Cold Zoom Diffusion


### Description
We develop Cold Zoom Diffusion: a deterministic way of training a diffusion model to produce MNIST digits by continually zooming into the image and extrapolating as our degradation strategy. We also train a standard Denoising Diffusion Probabilistic Model with linear and cosine schedules, and compare the sampling perfomance between the two. 


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

## Autogeneration tools:
Co-pilot was used in the following way:
- Autocompleting documentation for functions and for scripts
- Sometimes we used the autocomplete feature for standard script routines (although these almost always need fixing and are acting as templates) such as:
    - #plotting the data
    - :Allowed to autocomplete the plotting script.
- Understand error messages with "Explain this error" button next to the error message.
- General questions such as:

How can i change a function name and have it change throughout my workspace?  

Answer: In Visual Studio Code, you can use the "Rename Symbol" feature to change a function name throughout your workspace. Here are the steps:

Place your cursor on the function name you want to change.
Right-click the function name, and select "Rename Symbol" from the context menu. Alternatively, you can use the shortcut F2.
Enter the new function name and press Enter.
VS Code will automatically find and update all occurrences of the function name in your workspace. This includes not only the function definition, but also all places where the function is called.

Please note that this feature depends on the language support provided by the installed extensions. If you're working with a language that doesn't have good support, the "Rename Symbol" feature might not work


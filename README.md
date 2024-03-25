# M2: Application of Machine Learning Coursework
# Cold Zoom Diffusion


### Description
We develop Cold Zoom Diffusion: a deterministic way of training a diffusion model to produce MNIST digits by continually zooming into the image and extrapolating as our degradation strategy. We also train a standard Denoising Diffusion Probabilistic Model with linear and cosine schedules, and compare the sampling perfomance between the two.



### Installation
Dependencies required to run the project are listed in the `environment.yml` file. To install the necessary Conda environment from these dependencies, run:
```bash
conda env create -f environment.yml
```

Once the environment is created, activate it using:

```bash
conda activate m2
```

### File Contents
All scripts have a description of their usage at the top. They are broken into the following sections:



| Scipt                       | Usage
|----------------------------|---------------------------------|
| `train_ddpm.py`, <br> `train_zoom_bilinear.py`, <br> `train_zoom_nearest.py`                      | Training scripts for DDPM, Cold Zoom Diffusion with  bilinear and nearest interpolation |
| `plot_fid_and_samples.py

All the code was ran on a CPU: Ryzen 9, 16gb of RAM, GPU: NVIDIA RTX 3060



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

## ChatGPT usage:
ChatGPT 3.5 was used to:

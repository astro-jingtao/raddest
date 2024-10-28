# rarden 

The JAX-based Python implementation of the RARDEN (Regression for Astronomical data with Realistic Distributions, Errors and Non-linearity) algorithm.

## Installation

It is recommended to use a Python virtual environment to install and run `rarden`.
First, create a virtual environment:

```
conda create -n rarden
conda activate rarden
```

Install the `JAX` package following the instructions on the [official website](https://jax.readthedocs.io/en/latest/installation.html). Choose CPU or GPU version based on your hardware, the GPU version is recommended (if your machine supports it) for faster computation.

Then, install `rarden` using pip:

```
pip install git+https://github.com/astro-jingtao/rarden.git
```

Another package `ndtest` is required for using the `rarden` package. Install it using pip:  

```
pip install git+https://github.com/astro-jingtao/ndtest.git
```

Now. `rarden` should be ready to use.

## Usage

See the notebook in the `examples` folder for an example usage of `rarden`.

## Citation

If you use `rarden` in your research, please cite the following paper:

```
TBD
```

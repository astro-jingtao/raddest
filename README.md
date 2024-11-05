# garden 

The JAX-based Python implementation of the garden (reGression for Astronomical data with Realistic Distributions, Errors and Non-linearity) algorithm.

## Installation

It is recommended to use a Python virtual environment to install and run `garden`.
First, create a virtual environment:

```bash
conda create -n garden python=3.9 matplotlib ipykernel
conda activate garden
```

The `python==3.9` is the development version of `garden`, and the `python>3.9` should also work. If you find any issues with higher versions, please let me know by raising an issue.

The `python<3.9` is not supported due to the requirement of `optax`.

`matplotlib` and `ipykernel` are used for running the jupyter notebook example.

Install the `JAX` package following the instructions on the [official website](https://jax.readthedocs.io/en/latest/installation.html). Choose CPU or GPU version based on your hardware, the GPU version is recommended (if your machine supports it) for faster computation.

Then, install `garden` using pip:

```bash
pip install git+https://github.com/astro-jingtao/garden.git
```

Another package `ndtest` is required for using the `garden` package. Install it following the instructions in [astro-jingtao/ndtest](https://github.com/astro-jingtao/ndtest?tab=readme-ov-file#installation) (or [its gitee mirror](https://gitee.com/the-wind-is-still-like-snow/ndtest#installation) for the user have trouble with using github).


Now. `garden` should be ready to use.

## Usage

See the notebook in the `examples` folder for an example usage of `garden`.

You need `corner` to run these examples. If you don't have, install it by
```bash
pip install corner
```

## Citation

If you use `garden` in your research, please cite the following paper:

```
TBD
```

## Acknowledge

Tao Jing thanks Ce Sui, Kai Wang, Kangning Diao, Yangyao Chen, and Zechang Sun for various fruitful discussions about this project.

Tao Jing thanks Ce Sui, Junkai Zhang, and Qinglin Ma for early test of this package.

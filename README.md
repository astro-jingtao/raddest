# raddest 

The JAX-based Python implementation of the raddest (Regression for Astronomical Data with realistic Distributions, Errors and non-lineariTy) algorithm.

## Installation

It is recommended to use a Python virtual environment to install and run `raddest`.
First, create a virtual environment:

```bash
conda create -n raddest python=3.9 matplotlib ipykernel
conda activate raddest
```

The `python==3.9` is the development version of `raddest`, and the `python>3.9` should also work. If you find any issues with higher versions, please let me know by raising an issue.

The `python<3.9` is not supported due to the requirement of `optax`.

`matplotlib` and `ipykernel` are used for running the jupyter notebook example.

Install the `JAX` package following the instructions on the [official website](https://jax.readthedocs.io/en/latest/installation.html). Choose CPU or GPU version based on your hardware, the GPU version is recommended (if your machine supports it) for faster computation.

Then, install `raddest` using pip:

```bash
pip install git+https://github.com/astro-jingtao/raddest.git
```

Another package `ndtest` is required for using the `raddest` package. Install it following the instructions in [astro-jingtao/ndtest](https://github.com/astro-jingtao/ndtest?tab=readme-ov-file#installation) (or [its gitee mirror](https://gitee.com/the-wind-is-still-like-snow/ndtest#installation) for the user have trouble with using github).


Now. `raddest` should be ready to use.

## Usage

See the notebook in the `examples` folder for an example usage of `raddest`.

You need `corner` to run these examples. If you don't have, install it by
```bash
pip install corner
```

### Fitting with the energy distance

Besides the likelihood-based (`LogLinearML` and friends) and KS-grid-based (`LogLinearKS` and friends) methods, the two-sample energy distance can be used as the distribution-distance criterion, with two optimizers.

Grid search weighted by energy-test p-values (each grid point generates a synthetic sample and compares it to the observed one with a permutation test on the energy distance):

```python
from raddest.api.fitter import LogLinearEnergyPT

regressor = LogLinearEnergyPT(xflow, yflow, permutations=200)
params_mean, params_std, res_arr = regressor.fit(x_obs, y_obs,
                                                 params_list=params_list,
                                                 N=1000)
```

Gradient descent on the energy distance itself (the synthetic sample is generated differentiably from the parameters, and the energy distance to the observed sample is minimized with adam):

```python
from raddest.api.fitter import LogLinearEnergyGrad
import jax.numpy as jnp

regressor = LogLinearEnergyGrad(xflow, yflow)
params_hat, loss_history = regressor.fit(x_obs, y_obs,
                                         jnp.asarray([k0, b0, sig0]),
                                         N=1000)
```

Both regressors also provide `energy_test(params, x_obs, y_obs)`, which evaluates the fitted model against the observed data with a fresh synthetic sample and returns the energy distance and its p-value. Each energy-test evaluation costs `O((N + M)^2 * permutations)`; reduce `N` or `permutations` for large datasets.

## Claude skill

The package ships a Claude Agent Skill (`raddest-regression`) that guides an agent through choosing and running the fitting methods. After installing `raddest`, print the installed skills root and the deployment guide with:

```bash
python -m raddest.skills
```

See the printed `SKILL-DEPLOY.md` for how to link the skill into Claude Code.

## Citation

If you use `raddest` in your research, please cite the following paper:

```
@ARTICLE{2024arXiv241108747J,
       author = {{Jing}, Tao and {Li}, Cheng},
        title = "{Regression for Astronomical Data with Realistic Distributions, Errors and Non-linearity}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Astrophysics of Galaxies},
         year = 2024,
        month = nov,
          eid = {arXiv:2411.08747},
        pages = {arXiv:2411.08747},
          doi = {10.48550/arXiv.2411.08747},
archivePrefix = {arXiv},
       eprint = {2411.08747},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024arXiv241108747J},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

## Acknowledge

The implementation of multivariate Gauss Hermite quadrature is taken from https://github.com/arjungecon/QmUaArNkTet/blob/main/Final%20Assignment/PyCode/Code/ExpectationMultivariateNormal.py.

Tao Jing thanks Ce Sui, Kai Wang, Kangning Diao, Yangyao Chen, and Zechang Sun for various fruitful discussions about this project.

Tao Jing thanks Ce Sui, Junkai Zhang, and Qinglin Ma for early test of this package.

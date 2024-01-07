## How to use

### Functional programming

**Fit $p(x)$**

Here we use the `neglogplus` as pre-NF transformation.

```python
from nreg.x_xerr import nf_fit_trans_neglogplus, gen_xobs_trans_neglogplus

# fit the NF
flow, loss, params = nf_fit_trans_neglogplus(x_obs,
                        x_err,
                        optimizer=None,
                        n_pre=100,
                        n_train=500,
                        n_node=21,
                        verbose=False,
                        flow=None,
                        x_min=0,
                        x_max=15,
                        x_err_min=0,
                        x_err_max=5,
                        check_per_step=10,
                        early_stop=True,
                        max_stuck=5,
                        stuck_frac=0.05)
# generate data by using fitted NF
x_gen, x_err_gen, x_obs_gen = gen_xobs_trans_neglogplus(flow, 10000, 0, x_obs.max(), 0, x_err.max(), params, return_numpy=True)
```
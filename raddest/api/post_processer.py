from copy import deepcopy

import numpy as np
from ndtest.ksnd import ks2d2s
from raddest.api.fitter import FlowVI
from raddest.transfer import combine_transfers, XShift


def post_shift(x_obs, x_err, flow_vi: FlowVI, N=10000, shift_method='median'):

    flow_vi_old = flow_vi
    _, x_err_gen_old, x_obs_gen_old = flow_vi_old.sample(N)
    _, p_old = ks2d2s(x_obs_gen_old, x_err_gen_old, x_obs, x_err)

    flow_vi = deepcopy(flow_vi_old)
    x_shift_param = get_shift(x_obs, x_obs_gen_old, method=shift_method)
    transer_new = combine_transfers(XShift(), flow_vi_old.transfer)
    transfer_params_new = [x_shift_param] + list(flow_vi_old.transfer_params)

    flow_vi.transfer = transer_new
    flow_vi.transfer_params = transfer_params_new

    _, x_err_gen_new, x_obs_gen_new = flow_vi.sample(N)
    _, p_new = ks2d2s(x_obs_gen_new, x_err_gen_new, x_obs, x_err)

    return flow_vi if p_old < p_new else flow_vi_old


def get_shift(x_obs, x_obs_gen, method='median'):

    if method == 'median':
        return np.median(x_obs_gen) - np.median(x_obs)
    elif method == 'mean':
        return np.mean(x_obs_gen) - np.mean(x_obs)
    elif method == 'histgram':
        _range = np.percentile(x_obs, [25, 75])

        N_gen, edges_gen = np.histogram(x_obs_gen, bins=101, range=_range)
        cen_gen = (edges_gen[1:] + edges_gen[:-1]) / 2

        N_obs, edges_obs = np.histogram(x_obs, bins=101, range=_range)
        cen_obs = (edges_obs[1:] + edges_obs[:-1]) / 2

        return cen_gen[np.argmax(N_gen)] - cen_obs[np.argmax(N_obs)]

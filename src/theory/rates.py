# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from nnmt.lif.exp import _firing_rate_shift
from nnmt.lif._general import _mean_input, _std_input

__all__ = [
    "initial_rates_uniform",
    "solve",
]

def initial_rates_uniform(net_dict, trials, min, max, seed=None):
    """
    Draw uniformly distributed initial rates between min and max.
    Supports multiple trials.
    """
    # remove empty popultions
    N = net_dict['neuron_numbers'].sort_index()
    mask = N > 0
    N = N.loc[mask]
    # create uniform initial rates
    initial_rates = {}
    initial_rate_params = dict(size=N.shape[0], low=min, high=max)
    np.random.seed(seed)
    for trial in range(trials):
        initial_rates[trial] = np.random.uniform(**initial_rate_params)
    return pd.DataFrame(initial_rates, index=N.index)


def solve(net_dict, t, initial_rates):
    r"""
    Solves the auxiliary ODE system
      \dot{\nu}_i = -\nu_i + \phi(\mu_i, \sigma_i)
    with initial condition initial_rates at the times t.
    mu and sigma are determined by the recurrent input, hence a fixed point
    of the dynamics corresponds to a self-consistent solution.
    """
    # collect network parameters, make sure everything is sorted the same way
    N = net_dict['neuron_numbers'].sort_index()
    nu_ext = net_dict['rate_ext'].sort_index()  # 1/s
    K = net_dict['synapses_internal'].sort_index(axis=0).sort_index(axis=1)
    K = K.div(N, axis=0)
    K_ext = net_dict['synapses_external'].sort_index()
    K_ext = K_ext.div(N)
    J = net_dict['weights'].sort_index(axis=0).sort_index(axis=1)
    J_ext = net_dict['weights_ext'].sort_index()

    # remove empty populations
    mask = N > 0
    N = N.loc[mask]
    nu_ext = nu_ext.loc[mask]
    K = K.loc[mask, mask]
    K_ext = K_ext.loc[mask]
    J = J.loc[mask, mask]
    J_ext = J_ext.loc[mask]

    # collect neuron parameters
    lif_params_E = net_dict['neuron_params_E']
    lif_params_I = net_dict['neuron_params_I']
    ix_E = (slice(None), slice(None), 'E')
    ix_I = (slice(None), slice(None), 'I')
    tau_m = pd.Series(data=0., index=N.index)
    tau_m.loc[ix_E] = 1e-3*lif_params_E['tau_m']  # s
    tau_m.loc[ix_I] = 1e-3*lif_params_I['tau_m']  # s
    tau_syn = pd.DataFrame(data=0., index=N.index, columns=N.index)
    tau_syn.loc[ix_E, ix_E] = 1e-3*lif_params_E['tau_syn_ex']  # s
    tau_syn.loc[ix_E, ix_I] = 1e-3*lif_params_E['tau_syn_in']  # s
    tau_syn.loc[ix_I, ix_E] = 1e-3*lif_params_I['tau_syn_ex']  # s
    tau_syn.loc[ix_I, ix_I] = 1e-3*lif_params_I['tau_syn_in']  # s
    tau_syn_ext = pd.Series(data=0., index=N.index)
    tau_syn_ext.loc[ix_E] = 1e-3*lif_params_E['tau_syn_ex']  # s
    tau_syn_ext.loc[ix_I] = 1e-3*lif_params_I['tau_syn_ex']  # s
    t_ref = pd.Series(data=0., index=N.index)
    t_ref.loc[ix_E] = 1e-3*lif_params_E['t_ref']  # s
    t_ref.loc[ix_I] = 1e-3*lif_params_I['t_ref']  # s
    C_m = pd.DataFrame(data=0., index=N.index, columns=N.index)
    C_m.loc[ix_E, :] = lif_params_E['C_m']  # pF
    C_m.loc[ix_I, :] = lif_params_I['C_m']  # pF
    C_m_ext = pd.Series(data=0., index=N.index)
    C_m_ext.loc[ix_E] = lif_params_E['C_m']  # pF
    C_m_ext.loc[ix_I] = lif_params_I['C_m']  # pF
    V_th = pd.Series(data=0., index=N.index)
    V_th.loc[ix_E] = lif_params_E['V_th'] - lif_params_E['E_L']  # mV
    V_th.loc[ix_I] = lif_params_I['V_th'] - lif_params_I['E_L']  # mV
    V_r = pd.Series(data=0., index=N.index)
    V_r.loc[ix_E] = lif_params_E['V_reset'] - lif_params_E['E_L']  # mV
    V_r.loc[ix_I] = lif_params_I['V_reset'] - lif_params_I['E_L']  # mV

    # scale weights
    J *= 1e3*tau_syn/C_m  # mV
    J_ext *= 1e3*tau_syn_ext/C_m_ext  # mV

    #  ODE for (pseudotime) fixed point iteration
    def aux_ode(_, nu_in):
        mu = _mean_input(nu=nu_in, **input_prms)
        sig = _std_input(nu=nu_in, **input_prms)
        tau_eff = sig**2 / _std_input(nu=nu_in, **input_prms_scld)**2
        nu_out = _firing_rate_shift(V_0_rel=V_r, V_th_rel=V_th,
                                    mu=mu, sigma=sig,
                                    tau_m=tau_m, tau_r=t_ref, tau_s=tau_eff)
        return -nu_in + nu_out

    # solve auxilliary ODE with random initial conditions
    input_prms = dict(tau_m=tau_m, J=J, K=K, J_ext=np.diag(J_ext),
                      K_ext=np.diag(K_ext), nu_ext=nu_ext)
    input_prms_scld = dict(tau_m=tau_m, J=J, K=K/tau_syn, J_ext=np.diag(J_ext),
                           K_ext=np.diag(K_ext/tau_syn_ext), nu_ext=nu_ext)
    rates = {}
    for trial in initial_rates:
        sol_ivp = solve_ivp(fun=aux_ode, t_span=(t[0], t[-1]),
                            y0=initial_rates[trial], method='LSODA', t_eval=t)
        assert sol_ivp.success
        for i, (area, layer, pop) in enumerate(N.index):
            rates[(area, layer, pop, trial)] = sol_ivp.y[i, :]
    return pd.DataFrame(rates)

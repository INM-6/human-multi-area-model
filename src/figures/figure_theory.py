import warnings
from os.path import join as path_join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from ..network import networkDictFromDump
from ..theory.rates import solve, initial_rates_uniform

__all__ = [
    "load_network_and_simulation",
    "solve_theory",
    "plot_results",
    "meanfield_rate"
]

def load_network_and_simulation(net_hash, sim_hash):
    net_dict = networkDictFromDump(path_join('out', net_hash))
    folder_sim = path_join('out', net_hash, sim_hash)
    rates_sim = pd.read_pickle(path_join(folder_sim, 'rates.pkl'))
    rates_sim = rates_sim.sort_index()
    return net_dict, rates_sim

def solve_theory(net_dict, t_arr, min_rate_0, max_rate_0):
    initial_rates = initial_rates_uniform(net_dict=net_dict, trials=1, seed=123,
                                          min=min_rate_0, max=max_rate_0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rates = solve(net_dict=net_dict, t=t_arr, initial_rates=initial_rates)
    rates_theo = rates.iloc[-1, :].loc[(slice(None), slice(None), slice(None), 0)]
    return rates, rates_theo

def plot_results(rates, t_arr, rates_sim=None, rates_theo=None):
    plt.style.use('./misc/mplstyles/report_plots_master.mplstyle')
    fig = plt.figure(constrained_layout=True, figsize=(5.63, 1.5))
    label_prms = dict(fontsize=12, fontweight='bold', va='top', ha='right')
    gs = gridspec.GridSpec(1, 2, figure=fig)
    ax_ode = fig.add_subplot(gs[0, 0])
    ax_comp = fig.add_subplot(gs[0, 1])
    ax_comp.spines['top'].set_visible(False)
    ax_comp.spines['right'].set_visible(False)

    ode_mesh = ax_ode.pcolormesh(np.log10(rates).T, cmap='cividis')
    ode_cbar = fig.colorbar(ode_mesh, ax=ax_ode, ticks=[-2, -1, 0, 1], pad=0.0,
                            aspect=50, fraction=0.1)
    ax_ode.set_xticks([0, len(t_arr)//2, len(t_arr)])
    ax_ode.set_yticks([])
    ax_ode.set_xlabel('Pseudo-timesteps')
    ax_ode.set_ylabel('Population')
    ode_cbar.set_label(r'$\log_{10}$(rate) (spikes/s)')
    ax_ode.set_title('Theory convergence')
    ax_ode.text(s='A', transform=ax_ode.transAxes, x=-0., y=1.2, **label_prms)

    rate_min = min(rates_sim.min(), rates_theo.min())
    rate_max = max(rates_sim.max(), rates_theo.max())
    rate_corr = np.corrcoef(rates_sim, rates_theo)[0, 1]

    ax_comp.scatter(rates_sim, rates_theo, c='black')
    ax_comp.plot([rate_min, rate_max], [rate_min, rate_max], ':', c='silver')
    ax_comp.text(s=r'$\rho$'+f' = {rate_corr:.3f}', x=0.2, y=0.8,
                    transform=ax_comp.transAxes)
    ax_comp.set_xlabel('Rate sim. (spikes/s)')
    ax_comp.set_ylabel('Rate theo. (spikes/s)')
    ax_comp.set_xscale('log')
    ax_comp.set_yscale('log')
    ax_comp.set_title('Comparison to simulation')
    ax_comp.text(s='B', transform=ax_comp.transAxes, x=-0.15, y=1.2, **label_prms)

    fig.savefig('figures/figure_mean_field.pdf')
    plt.show()

def meanfield_rate(outpath, net_hash, sim_hash=None):
    t_arr = 0.1 * np.arange(100)
    min_rate_0, max_rate_0 = 0., 10.

    net_dict, rates_sim = None, None
    if sim_hash:
        net_dict, rates_sim = load_network_and_simulation(net_hash, sim_hash)
    else:
        net_dict = networkDictFromDump(path_join(outpath, net_hash))

    rates, rates_theo = solve_theory(net_dict, t_arr, min_rate_0, max_rate_0)
    if sim_hash:
        plot_results(rates, t_arr, rates_sim, rates_theo)
    else:
        print(f'Average theoretical rate: {rates_theo.mean()}')

    return rates_theo

if __name__ == "__main__":
    net_hash = '8757444d0e17c59775b273562693e60b'
    sim_hash = 'b866fc17e887c17e2af2d8b06be6c9b5'  # Set to None if you don't want to use simulated values
    meanfield_rate(outpath, net_hash, sim_hash)
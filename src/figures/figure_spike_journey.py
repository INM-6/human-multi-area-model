from os.path import join as path_join
from itertools import product, combinations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.transforms

import networkx as nx

from ..network import networkDictFromDump
from ..data_loader.dk_fullnames_to_shortnames import dk_full_to_short

# set params &  hashes
dt = 0.1
step_min, step_max = 20000, 21000

net_hash_orig_gs = '90523c45dfad8e5bacb2eaf4d2196f76'  # groundstate
net_hash_pert_gs = 'a80f3ffb485f0d19610312bf2195b89b'  # groundstate

net_hash_orig_bf = '8c49a09f51f44fbb036531ce0719b5ba'  # best fMRI
net_hash_pert_bf = '251f101af71db37db323cd889a89103a'  # best fMRI

sim_hash = 'a9a78923cdc29bfa5cb4cad82f30e5bb'
ana_hash = '386fd340a9cb17c9d7b1ced530448025'

# outpath = 'out'
outpath = os.path.join(os.getcwd(), 'out')

def load_diffs(net_hash_orig, net_hash_pert):
    """Calculate rate differences for the two hashes"""
    folder_orig = path_join(outpath, net_hash_orig, sim_hash)
    folder_pert = path_join(outpath, net_hash_pert, sim_hash)
    # load data
    rates_orig = pd.read_pickle(path_join(
        folder_orig, ana_hash, 'rate_histogram.pkl'))
    rates_orig = pd.DataFrame(rates_orig.sort_index().to_dict())
    rates_pert = pd.read_pickle(path_join(
        folder_pert, ana_hash, 'rate_histogram.pkl'))
    rates_pert = pd.DataFrame(rates_pert.sort_index().to_dict())
    # binary difference matrix
    rates_diff = rates_orig - rates_pert
    rates_diff = rates_diff[rates_diff.index >= step_min]
    rates_diff = rates_diff[rates_diff.index <= step_max]
    rates_diff = np.heaviside(rates_diff.T.abs(), 0)
    # calculate index of first difference
    first_diff = {}
    for ind, rate_diff in rates_diff.iterrows():
        first_diff[ind] = np.argmax(np.abs(rate_diff) > 0)
    first_diff = pd.Series(first_diff).sort_values()
    first_diff[first_diff == step_min] = step_max + 1  # outside of window
    first_diff = dt*(first_diff - step_min)

    return rates_diff, first_diff


def sload(net_hashes):
    """
    Assert that all returns are the same in all net_hashes and return them
    """
    for net_hash_A, net_hash_B in combinations(net_hashes, r=2):
        net_dict_A = networkDictFromDump(path_join(outpath, net_hash_A))
        net_dict_B = networkDictFromDump(path_join(outpath, net_hash_B))
        assert(net_dict_A['area_list'] == net_dict_B['area_list'])
        assert(np.isclose(net_dict_A['delay_e'], net_dict_B['delay_e']))
        assert(np.isclose(net_dict_A['delay_i'], net_dict_B['delay_i']))
        assert(np.allclose(net_dict_A['delay_cc'], net_dict_B['delay_cc']))
        assert(np.allclose(net_dict_A['neuron_numbers'],
                           net_dict_B['neuron_numbers']))
        assert(np.allclose(net_dict_A['synapses_internal'],
                           net_dict_B['synapses_internal']))
        pert_pop_A = net_dict_A['spike_time'].dropna().index.values
        pert_pop_B = net_dict_B['spike_time'].dropna().index.values
        if pert_pop_A and pert_pop_B:
            assert(pert_pop_A == pert_pop_B)
            assert(len(pert_pop_A) == 1)
            pert_pop = pert_pop_A[0]

    # all values equal, proceed with arbitrary net_dict
    area_list = net_dict_A['area_list']
    neurons = net_dict_A['neuron_numbers']
    synapses = net_dict_A['synapses_internal']
    delay_e = net_dict_A['delay_e']
    delay_i = net_dict_A['delay_i']
    delays_cc = net_dict_A['delay_cc']
    return area_list, neurons, synapses, delay_e, delay_i, delays_cc, pert_pop


area_list, neurons, synapses, delay_e, delay_i, delays_cc, pert_pop = sload([
    net_hash_orig_gs, net_hash_pert_gs, net_hash_orig_bf, net_hash_pert_bf
])

# create delay matrix
neurons = neurons[neurons > 0]
synapses = synapses.loc[neurons.index, neurons.index]
synapses = synapses.sort_index(axis=0).sort_index(axis=1)
delays_cc = delays_cc.sort_index(axis=0).sort_index(axis=1)
delay_matrix = synapses.astype(bool).astype(float)
for source, target in product(area_list, area_list):
    if source == target:
        delay_matrix.loc[
            (target, slice(None), slice(None)),
            (source, slice(None), 'E')
        ] *= delay_e
        delay_matrix.loc[
            (target, slice(None), slice(None)),
            (source, slice(None), 'I')
        ] *= delay_i
    else:
        delay_matrix.loc[
            (target, slice(None), slice(None)),
            (source, slice(None), slice(None))
        ] *= delays_cc.loc[target, source]

# compute shortest path based on mean delays
delay_graph = nx.from_numpy_array(delay_matrix.values.T,  # A_ij from i to j
                                  create_using=nx.DiGraph)
dijkstra_pl = nx.all_pairs_dijkstra_path_length(delay_graph)
dist_matrix = pd.DataFrame(data=dict(dijkstra_pl))
dist_matrix.index = delay_matrix.index
dist_matrix.columns = delay_matrix.columns
# compute shortest path in steps
shortest_pl = nx.all_pairs_shortest_path_length(delay_graph)
steps_matrix = pd.DataFrame(data=dict(shortest_pl))
steps_matrix.index = delay_matrix.index
steps_matrix.columns = delay_matrix.columns
# distance from perturbation
dist_pert = dist_matrix.loc[(slice(None), slice(None), slice(None)),
                            pert_pop]
steps_pert = steps_matrix.loc[(slice(None), slice(None), slice(None)),
                              pert_pop]

# compute rate diffs
rates_diff_gs, first_diff_gs = load_diffs(net_hash_orig_gs, net_hash_pert_gs)
rates_diff_bf, first_diff_bf = load_diffs(net_hash_orig_bf, net_hash_pert_bf)
# reduce first_diff to area-level
first_diff_area_gs = first_diff_gs.groupby(level=0).min().sort_values()
first_diff_area_bf = first_diff_bf.groupby(level=0).min().sort_values()
# use area-level min in first_diff
first_diff_min_gs = first_diff_gs.groupby(level=0).transform('min')
first_diff_min_gs = first_diff_min_gs.sort_index()
first_diff_min_bf = first_diff_bf.groupby(level=0).transform('min')
first_diff_min_bf = first_diff_min_bf.sort_index()

# initialize plot
plt.style.use('misc/mplstyles/report_plots_master.mplstyle')
fig = plt.figure(constrained_layout=True, figsize=(5.63, 5.5))
gs = gridspec.GridSpec(20, 20, figure=fig)
ax_delay = fig.add_subplot(gs[-3:, 10:])
ax_steps = fig.add_subplot(gs[-3:, :10])
ax_pert_gs = fig.add_subplot(gs[:-4, :8])
ax_pert_bf = fig.add_subplot(gs[:-4, 8:16])
ax_time = fig.add_subplot(gs[:-4, 16:])
for ax in [ax_delay, ax_steps, ax_time]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
label_prms = dict(fontsize=12, fontweight='bold', va='top', ha='right')
colors_state = {'groundstate': '#DDAA33', 'bestfit': '#004488'}

# calculate ticks & labels
order = first_diff_area_gs.index
assert(rates_diff_gs.index.tolist() == rates_diff_bf.index.tolist())
pops = rates_diff_gs.reindex(order, level=0).index.tolist()
ticks = []
areas = []
for i, pop in enumerate(pops):
    if pops[i-1][0] != pop[0]:
        ticks.append(i)
        areas.append(pop[0])
areas = [dk_full_to_short[x] for x in areas]

# plot
cmap = plt.cm.summer
nbins = 25
_, _, patches = ax_delay.hist(dist_matrix.values.ravel(), bins=nbins, log=True)
for i, p in enumerate(patches):
    plt.setp(p, 'facecolor', cmap(i/nbins))
ax_delay.set_xlim(0)
ax_delay.set_xlabel('Dijkstra path length [ms]')
ax_delay.set_ylabel('count')
ax_delay.set_title('Distance (mean delay)')
ax_delay.text(s='E', transform=ax_delay.transAxes, x=-0.05, y=1.4,
              **label_prms)

bins = np.arange(8) - 0.5
ax_steps.hist(steps_matrix.values.ravel(), bins=bins, color='black', log=True,
              rwidth=0.8)
ax_steps.set_xlabel('shortest path length')
ax_steps.set_title('Distance (steps)')
ax_steps.text(s='D', transform=ax_steps.transAxes, x=-0.05, y=1.4,
              **label_prms)

axs = [ax_pert_gs, ax_pert_bf]
diffs = [rates_diff_gs, rates_diff_bf]
labels = ['A', 'B']
titles = ['Spike Propagation (Base version)', 'Spike Propagation (Best-fit version)']
for ax_pert, rates_diff, label, title in zip(axs, diffs, labels, titles):
    rates_diff[rates_diff == 0] = np.nan
    dist_pert_reindex = dist_pert.loc[rates_diff.index]
    rates_diff_dist = rates_diff.multiply(dist_pert_reindex, axis=0)
    rates_diff_dist = rates_diff_dist.reindex(order, level=0)
    cmesh_pert = ax_pert.pcolormesh(rates_diff_dist, cmap=cmap)
    ax_pert.set_yticks(ticks)
    ax_pert.text(s=label, transform=ax_pert.transAxes, x=-0.05, y=1.055,
                 **label_prms)
    ax_pert.set_title(title)
    ax_pert.invert_yaxis()
    ax_pert.set_xlim(0, step_max-step_min)
    ax_pert.set_xticks((0, (step_max-step_min)/2, step_max-step_min))
    ax_pert.set_xticklabels((0, dt*(step_max-step_min)/2,
                            dt*(step_max-step_min)))
    ax_pert.set_xlabel('time [ms]')
ax_pert_gs.set_yticklabels(areas)
offset = matplotlib.transforms.ScaledTranslation(0, -1/72, fig.dpi_scale_trans)
# move label PC down to avoid overlap
for label in ax_pert_gs.yaxis.get_majorticklabels():
    if label.get_text() == 'PC':
        label.set_transform(label.get_transform() + offset)
ax_pert_bf.set_yticklabels([])
ax_pert_gs.set_ylabel('population')

ax_time.plot(first_diff_min_gs.reindex(order, level=0), np.arange(len(pops)),
             's', label='base version', color=colors_state['groundstate'])
ax_time.plot(first_diff_min_bf.reindex(order, level=0), np.arange(len(pops)),
             's', label='best-fit version', color=colors_state['bestfit'])

print('gs mean:', first_diff_min_gs.mean(), 'gs std:', first_diff_min_gs.std())
print('bf mean:', first_diff_min_bf.mean(), 'bf std:', first_diff_min_bf.std())
ax_time.set_yticks(ticks)
ax_time.set_yticklabels([])
ax_time.set_ylim(0, rates_diff_dist.shape[0])
ax_time.invert_yaxis()
ax_time.set_xlim(0, 75)
ax_time.set_xlabel('first pert. [ms]')
ax_time.set_title('First Perturbation')
ax_time.legend(loc='upper right', frameon=False, handletextpad=0.)
ax_time.text(s='C', transform=ax_time.transAxes, x=-0.05, y=1.055,
             **label_prms)

fig.savefig('figures/figure_spike_journey.pdf')
fig.savefig('figures/figure_spike_journey.png')

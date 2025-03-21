import random
import numpy as np
import pandas as pd
import os
from os.path import join as p_join

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


colors = {'E': '#4c72b0ff', 'I': '#c44e52ff'}

raster_fraction = 0.025
raster_low = 4000.0
raster_high = 5000.0

roman_to_arabic_numerals = {
        'II/III': '2/3',
        'IV': '4',
        'V': '5',
        'VI': '6',
        }

random.seed(1234)


outpath = os.path.join(os.getcwd(), 'out/')
net_folder = p_join(outpath, '8c49a09f51f44fbb036531ce0719b5ba')  # bestfit
sim_folder = p_join(net_folder, '4772f0b020c9f3310f4096a6db758343')
ana_folder = p_join(sim_folder, '7ebd64b6b9a95c3d8da8cf3af85e9985')

# ========== Load ==========
popGids = pd.read_pickle(p_join(sim_folder, 'population_GIDs.pkl'))
spikes = pd.read_pickle(p_join(sim_folder, 'spikes.pkl'))

areas = list(popGids.index.get_level_values(0).unique())

# ========== Plot ==========
plt.style.use('./misc/mplstyles/report_plots_master.mplstyle')

for num_fig in range(3):
    fig = plt.figure(constrained_layout=True, figsize=(5.63, 9))
    label_prms = dict(fontsize=12, fontweight='bold', va='top', ha='right')
    gs = gridspec.GridSpec(3, 4, figure=fig)
    axs_raster = []
    raster_areas = areas[num_fig*12:(num_fig+1)*12]
    for i, area in enumerate(raster_areas):
        axs_raster.append(fig.add_subplot(gs[i//4, i%4]))
        axs_raster[i].spines['top'].set_visible(False)
        axs_raster[i].spines['right'].set_visible(False)

    # raster plots
    ms_to_s = 1e-3
    for ax, area in zip(axs_raster, raster_areas):
        ind = []
        names = []
        gid_norm = 0
        for (layer, pop), sts in spikes.loc[area].iteritems():
            layer_roman = roman_to_arabic_numerals[layer]
            # Random shuffle spiketrains in place
            random.shuffle(sts)

            # Real population size, not all neurons spiked. Thus take the
            # fraction from this value.
            popGid_alp = popGids.loc[area, layer, pop]
            pop_size = popGid_alp.maxGID - popGid_alp.minGID + 1
            # Fraction of total number of neurons
            no_sts = int(raster_fraction * pop_size)
            # Fraction of neurons that actually spiked
            frac_spiking = len(sts) / pop_size

            # y label position and name
            ind.append(- int(no_sts / 2) + gid_norm)
            name = ' '.join([layer_roman, pop])
            names.append(name)

            j = 0
            # Loop as many times as we have spike trains
            for _ in range(no_sts):
                gid_norm = gid_norm - 1
                # Decide whether spiketrain contains spikes
                if random.random() < frac_spiking:
                    st = sts[j]
                    j += 1
                    filtered_st = st[st > raster_low]
                    filtered_st = filtered_st[filtered_st < raster_high]
                    if len(filtered_st) > 0:
                        ax.plot(
                            filtered_st * ms_to_s,
                            gid_norm * np.ones_like(filtered_st),
                            colors[pop],
                            marker='.',
                            markersize=2,
                            linestyle="None"
                        )

        ax.axis([raster_low * ms_to_s, raster_high * ms_to_s, gid_norm, 0])
        ax.set_xlabel('Time (s)')
        ax.set_yticks(ind)
        ax.set_yticklabels(names)
        ax.set_title(area)

    # save figure
    fig.savefig(f'figures/figure_spike_appendix_{num_fig}.pdf')

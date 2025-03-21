import os
import random
import numpy as np
import pandas as pd
from os.path import join as p_join

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns


colors = {'E': '#4c72b0ff', 'I': '#c44e52ff'}

raster_areas = ['caudalanteriorcingulate', 'pericalcarine', 'fusiform']
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


for state in ['groundstate', 'bestfit']:
    name_extension = '_lichtman_chiI2_different_seed'
    outpath = os.path.join(os.getcwd(), 'out/')
    if state == 'groundstate':
        net_folder = p_join(outpath, '90523c45dfad8e5bacb2eaf4d2196f76')  # groundstate
    elif state == 'bestfit':
        net_folder = p_join(outpath, '8c49a09f51f44fbb036531ce0719b5ba')  # bestfit
    sim_folder = p_join(net_folder, '4772f0b020c9f3310f4096a6db758343')
    ana_folder = p_join(sim_folder, '7ebd64b6b9a95c3d8da8cf3af85e9985')

    # ========== Load ==========
    popGids = pd.read_pickle(p_join(sim_folder, 'population_GIDs.pkl'))
    spikes = pd.read_pickle(p_join(sim_folder, 'spikes.pkl'))
    rate = pd.read_pickle(p_join(sim_folder, 'rates.pkl'))
    lv = pd.read_pickle(p_join(sim_folder, 'lv.pkl'))
    cv = pd.read_pickle(p_join(sim_folder, 'cv_isi.pkl'))
    cc = pd.read_pickle(p_join(sim_folder, 'cc.pkl'))


    # ========== Plot ==========
    plt.style.use('./misc/mplstyles/report_plots_master.mplstyle')
    fig = plt.figure(constrained_layout=True, figsize=(5.63, 3.5))
    label_prms = dict(fontsize=12, fontweight='bold', va='top', ha='right')
    gs = gridspec.GridSpec(3, 4, figure=fig)
    ax_raster1 = fig.add_subplot(gs[:, 0])
    ax_raster1.spines['top'].set_visible(False)
    ax_raster1.spines['right'].set_visible(False)
    ax_raster2 = fig.add_subplot(gs[:, 1])
    ax_raster2.spines['top'].set_visible(False)
    ax_raster2.spines['right'].set_visible(False)
    ax_raster3 = fig.add_subplot(gs[:, 2])
    ax_raster3.spines['top'].set_visible(False)
    ax_raster3.spines['right'].set_visible(False)
    ax_rates = fig.add_subplot(gs[0, 3])
    ax_rates.spines['top'].set_visible(False)
    ax_rates.spines['right'].set_visible(False)
    ax_cv = fig.add_subplot(gs[1, 3])
    ax_cv.spines['top'].set_visible(False)
    ax_cv.spines['right'].set_visible(False)
    ax_cc = fig.add_subplot(gs[2, 3])
    ax_cc.spines['top'].set_visible(False)
    ax_cc.spines['right'].set_visible(False)

    # raster plots
    ms_to_s = 1e-3
    axs_raster = [ax_raster1, ax_raster2, ax_raster3]
    raster_labels = ['A', 'B', 'C']
    for ax, area, label in zip(axs_raster, raster_areas, raster_labels):
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
        ax.text(s=label, transform=ax.transAxes, x=-0.2, y=1.06, **label_prms)

    # boxplots
    axs_boxplots = [ax_rates, ax_cv, ax_cc]
    data_boxplots = [rate, cv, cc]
    boxplots_labels = ['D', 'E', 'F']
    for ax, data, label in zip(axs_boxplots, data_boxplots, boxplots_labels):
        # reorder Series into DataFrame
        area = np.unique(data.index.get_level_values(0))
        layer = np.unique(data.index.get_level_values(1))
        pop_type = np.unique(data.index.get_level_values(2))
        multi_index = pd.MultiIndex.from_product([layer, pop_type])
        ind = [' '.join(i) for i in multi_index.tolist()]
        names = []
        for i, layer_name in enumerate(ind):
            l_, p_ = layer_name.split(' ');
            names.append(' '.join((roman_to_arabic_numerals[l_], p_)))
        data_lp = pd.DataFrame(data=np.nan, index=area, columns=ind)
        for (a, l, p), r in data.iteritems():
            data_lp.loc[a, l+' '+p] = r

        boxplot = sns.boxplot(data=data_lp, orient='h', ax=ax, saturation=1,
                              width=0.5, fliersize=2.5, color='k')
        col = [colors['E'], colors['I']]
        for i in range(len(ind)):
            mybox = boxplot.artists[i]
            mybox.set_facecolor(col[i % 2])
        # ax_rates.set_ylabel('Population')
        ax.text(s=label, transform=ax.transAxes, x=-0.1, y=1.25, **label_prms)
        # Print the extension of the whiskers
        lower = []
        upper = []
        for name, x in data_lp.iteritems():
            dat = x.dropna().values
            median = np.median(dat)
            upper_quartile = np.percentile(dat, 75)
            lower_quartile = np.percentile(dat, 25)
            iqr = upper_quartile - lower_quartile
            upper_whisker = dat[dat<=upper_quartile+1.5*iqr].max()
            lower_whisker = dat[dat>=lower_quartile-1.5*iqr].min()
            lower.append(lower_whisker)
            upper.append(upper_whisker)
        print('state:', state, 'label:', label, 'lowest whisker:', round(min(lower), 1))
        print('state:', state, 'label:', label, 'highest whisker:', round(max(upper), 1))
        ax.set_yticklabels(names)
    ax_rates.set_xlim(0)
    ax_rates.set_xlabel('Firing rate (spikes/s)')
    ax_cv.set_xlim(0)
    ax_cv.set_xlabel('CV interspike interval')
    ax_cc.set_xlabel('Correlation coefficient')

    # save figure
    fig.savefig(f'figures/figure_spike_statistics_{state}{name_extension}.pdf')

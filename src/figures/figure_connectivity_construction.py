import os
import numpy as np
import pandas as pd
from scipy.special import ndtr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from ..network import networkDictFromDump
from ..data_loader.microcircuit import p as p_PD
from ..data_loader.synapse_cellbody_probability import mohan
from ..data_loader.dk_fullnames_to_shortnames import dk_full_to_short


has_macaque_data = True
try:
    macaque_data = pd.read_pickle('experimental_data/macaque/macaque_data_merged.pkl')
except FileNotFoundError:
    print('WARNING: Did not find experimental_data/macaque/macaque_data_merged.pkl so the \
           data in the background of panel D will be missing in the figure. To \
           include the data please check if you have the necessary files in \
           experimental_data/macaque/, then run the script experimental_data/macaque/preprocessing.py')
    has_macaque_data = False
    exit(1)
net_dict = networkDictFromDump(os.path.join(os.getcwd(), 'out/8c49a09f51f44fbb036531ce0719b5ba/'))

roman_to_arabic_numerals = {
        'I': '1',
        'II/III': '2/3',
        'IV': '4',
        'V': '5',
        'VI': '6',
        }

# ========== Preprocess ==========

# Microcircuit connectivity probabilities
p_PD[p_PD < 1e-9] = np.nan

# Synapse to cell body probability
mohan = mohan.groupby('layer').sum()
synapse_layer = mohan.columns.tolist()
soma_layer = mohan.index.tolist()
mohan_extended = pd.DataFrame(data=np.nan, index=[synapse_layer[0]],
                              columns=synapse_layer).append(mohan)
synapse_layer_label = [roman_to_arabic_numerals[layer] for layer in synapse_layer]
soma_layer_label = [roman_to_arabic_numerals[layer] for layer in soma_layer]

# DTI data
conn = net_dict['NOS']
conn = conn.sort_index(axis=0).sort_index(axis=1)
log_conn = np.log10(conn)

# SLN data
syn_vek = net_dict[
    'synapses_internal'
].groupby(level=['area'], axis=0).sum().groupby(level=['area'], axis=1).sum()
sln_vek = net_dict['SLN'][syn_vek > 0]
for x in sln_vek.index:
    sln_vek.loc[x][x] = np.nan

# SLN fit
if has_macaque_data:
    macaque_logratio = macaque_data['DENS_LOGRATIO']
    macaque_sln = macaque_data['SLN']
a0 = net_dict['synapsenumbers_params']['a0']
a1 = net_dict['synapsenumbers_params']['a1']
x_arr = np.linspace(-2, 2, 1001)
fit_arr = ndtr(a0 + a1*x_arr)

# Mesoconectome
synapses = net_dict['synapses_internal']
synapses = synapses.sort_index(axis=0).sort_index(axis=1)
log_synapses = np.log10(synapses)
log_synapses[log_synapses == -np.inf] = np.nan

# ========== Plot ==========

plt.style.use('./misc/mplstyles/report_plots_master.mplstyle')
fig = plt.figure(constrained_layout=True, figsize=(5.63, 6.5))
label_prms = dict(fontsize=12, fontweight='bold', va='top', ha='right')
gs = gridspec.GridSpec(8, 2, figure=fig)
ax_PD14 = fig.add_subplot(gs[:3, 0])
ax_DTI = fig.add_subplot(gs[:3, 1])
ax_S2S = fig.add_subplot(gs[3:5, 0])
ax_FIT = fig.add_subplot(gs[3:5, 1])
ax_DIR = fig.add_subplot(gs[5:, 0])
ax_MESO = fig.add_subplot(gs[5:, 1])

# microcircuit
assert p_PD.index.tolist() == p_PD.columns.tolist()
pops = p_PD.index.tolist()
pops = [roman_to_arabic_numerals[x[0]] + ' ' + x[1] for x in pops]
im = ax_PD14.pcolor(p_PD, vmin=0, vmax=p_PD.max().max(), cmap='YlGn')
ax_PD14.set_xticks(np.arange(0.5, len(pops), 1))
ax_PD14.set_xticklabels(pops, rotation='vertical')
ax_PD14.set_yticks(np.arange(0.5, len(pops), 1))
ax_PD14.set_yticklabels(pops)
ax_PD14.axis('square')
ax_PD14.invert_yaxis()
ax_PD14.set_ylabel('Target')
ax_PD14.set_xlabel('Source')
ax_PD14.set_title('Intrinsic connectivity')
cbar = fig.colorbar(im, ax=ax_PD14, shrink=0.7)
cbar.set_ticks(np.arange(0, p_PD.max().max(), 0.2))
cbar.set_label('average pairwise synapses', rotation=270, labelpad=15)
ax_PD14.text(s='A', transform=ax_PD14.transAxes, x=-0.2, y=1.2, **label_prms)

# DTI
assert log_conn.index.tolist() == log_conn.columns.tolist()
areas = log_conn.index.tolist()
areas = [dk_full_to_short[x] for x in areas]
im = ax_DTI.pcolor(log_conn, vmin=log_conn[log_conn > -np.inf].min().min(),
                   vmax=log_conn.max().max(), cmap='YlGn')
ax_DTI.set_xticks(np.arange(1.5, len(areas), 2))
ax_DTI.set_xticklabels(areas[1::2], rotation='vertical')
ax_DTI.set_yticks(np.arange(0.5, len(areas), 2))
ax_DTI.set_yticklabels(areas[::2])
ax_DTI.axis('square')
ax_DTI.invert_yaxis()
ax_DTI.set_ylabel('Target')
ax_DTI.set_xlabel('Source')
ax_DTI.set_title('Cortico-cortical connectivity (DTI)')
cbar = fig.colorbar(im, ax=ax_DTI, shrink=0.7)
cbar.set_ticks(np.arange(0, log_conn.max().max(), 1))
cbar.set_label(r'$\log_{10}$'+'(# streamlines)', rotation=270, labelpad=15)
ax_DTI.text(s='B', transform=ax_DTI.transAxes, x=-0.2, y=1.2, **label_prms)

# Mohan data
im = ax_S2S.pcolor(mohan, vmin=0, vmax=1, cmap='YlGn')
ax_S2S.set_xticks(np.arange(0.5, len(synapse_layer_label), 1))
ax_S2S.set_xticklabels(synapse_layer_label, rotation='vertical')
ax_S2S.set_yticks(np.arange(0.5, len(soma_layer_label), 1))
ax_S2S.set_yticklabels(soma_layer_label)
ax_S2S.set_xlabel('Synapse')
ax_S2S.set_ylabel('Soma')
ax_S2S.invert_yaxis()
cbar = fig.colorbar(im, ax=ax_S2S, shrink=0.7, aspect=6.5)
cbar.set_ticks([0, 1])
cbar.set_label('Probability', rotation=270, labelpad=15)
ax_S2S.set_title('Synapse-to-soma probability')
ax_S2S.text(s='C', transform=ax_S2S.transAxes, x=-0.2, y=1.4, **label_prms)

# Fit
if has_macaque_data:
    ax_FIT.plot(macaque_logratio, macaque_sln, '.',
                markersize=4, color='silver', label='data')
ax_FIT.plot(x_arr, fit_arr, '-', color='black', label='fit')
ax_FIT.set_xlim(x_arr[0], x_arr[-1])
ax_FIT.set_ylim(0, 1)
ax_FIT.set_xlabel(r'$\log_{10}(\rho_A / \rho_B)$')
ax_FIT.set_ylabel('SLN')
ax_FIT.legend(loc='lower left', frameon=True)
ax_FIT.set_title('SLN fit (macaque)')
ax_FIT.text(s='D', transform=ax_FIT.transAxes, x=-0.2, y=1.4, **label_prms)

# Directionality
areas = [dk_full_to_short[name] for name in sln_vek.columns]
im = ax_DIR.pcolormesh(sln_vek, vmin=0, vmax=1, cmap='YlGn')
ax_DIR.set_xticks(np.arange(1.5, len(areas), 2))
ax_DIR.set_xticklabels(areas[1::2], rotation='vertical')
ax_DIR.set_yticks(np.arange(0.5, len(areas), 2))
ax_DIR.set_yticklabels(areas[::2])
ax_DIR.axis('square')
ax_DIR.invert_yaxis()
cbar = fig.colorbar(im, ax=ax_DIR, shrink=0.7, pad=-.243)
cbar.set_ticks([0, 1])
cbar.set_label('SLN', rotation=270, labelpad=5)
ax_DIR.set_title('SLN prediction (human)')
ax_DIR.text(s='E', transform=ax_DIR.transAxes, x=-0.2, y=1.2, **label_prms)

# Mesoconnectome
assert log_synapses.index.tolist() == log_synapses.columns.tolist()
pops = log_synapses.index.tolist()
ticks = []
areas = []
for i, pop in enumerate(pops):
    if pops[i-1][0] != pop[0]:
        ticks.append(i)
        areas.append(pop[0])
areas = [dk_full_to_short[x] for x in areas]
im = ax_MESO.pcolor(log_synapses,
                    vmin=log_synapses[log_synapses > -np.inf].min().min(),
                    vmax=log_synapses.max().max(), cmap='YlGn')
ax_MESO.set_xticks(ticks[1::2])
ax_MESO.set_xticklabels(areas[1::2], rotation='vertical', ha='left')
ax_MESO.set_yticks(ticks[::2])
ax_MESO.set_yticklabels(areas[::2], fontsize=6, va='top')
ax_MESO.axis('square')
ax_MESO.invert_yaxis()
ax_MESO.set_ylabel('Target')
ax_MESO.set_xlabel('Source')
cbar = fig.colorbar(im, ax=ax_MESO, shrink=0.7)
cbar.set_ticks(np.arange(0, log_synapses.max().max(), 1))
cbar.set_label(r'$\log_{10}$'+'(# synapses)', rotation=270, labelpad=15)
ax_MESO.set_title('Mesoconnectome')
ax_MESO.text(s='F', transform=ax_MESO.transAxes, x=-0.2, y=1.2, **label_prms)

fig.savefig('figures/figure_connectivity_construction.pdf')

import os
from os.path import join as path_join
from itertools import product
import yaml
import numpy as np
import pandas as pd
from scipy.stats import norm, shapiro, linregress

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import svgutils.transform as sg

from ..network import networkDictFromDump


pval_min = 0.05
target_min_outdeg = 100
work_dir = "figures/tmp"
figname = "figure_connectivity_validation"
net_dict = networkDictFromDump(os.path.join(os.getcwd(), 'out/8c49a09f51f44fbb036531ce0719b5ba/'))

# ========== Preprocess ==========
area_list = net_dict['area_list']
NN_orig = net_dict['neuron_numbers']
NN = NN_orig[NN_orig > 0]
NN = NN.sort_index()
synapses_orig = net_dict['synapses_internal']
synapses = synapses_orig.loc[NN.index, NN.index]
synapses = synapses.sort_index(axis=0).sort_index(axis=1)
directionality = net_dict['directionality']
directionality = directionality.sort_index(axis=0).sort_index(axis=1)
interarea_speed = net_dict['network_params']['delay_params']['interarea_speed']
distance = net_dict['delay_cc'] * interarea_speed
distance = distance.sort_index(axis=0).sort_index(axis=1)

# Community structure
synapses_area = synapses.groupby(
    'area', axis='index'
).sum().groupby(
    'area', axis='columns'
).sum()

# Synapses
log_synapses_flat = np.ravel(np.log10(synapses))
log_synapses_flat = log_synapses_flat[np.isfinite(log_synapses_flat)]
log_synapses_mean = np.mean(log_synapses_flat)
log_synapses_std = np.std(log_synapses_flat)
teststat_syn, pvalue_syn = shapiro(log_synapses_flat)
print(f'Synapse density lognormal with p={pvalue_syn}')
if pvalue_syn < pval_min:
    print('  rejected by Shapiro-Wilk test')

# EDR
synapses_area_flat = np.ravel(synapses_area)
dist_flat = np.ravel(distance)
synapses_area_flat = synapses_area_flat[dist_flat > 0]
dist_flat = dist_flat[dist_flat > 0]
dist_flat = dist_flat[synapses_area_flat > 0]
synapses_area_flat = synapses_area_flat[synapses_area_flat > 0]
edr_res = linregress(dist_flat, np.log10(synapses_area_flat))
edr_slope, edr_intercept, _, _, _ = edr_res
edr_decay = - 1 / (np.log(10) * edr_slope)
print(f'EDR decay constant {edr_decay} mm')

# Outdegree
outdeg = synapses.div(NN, axis='columns')
directionality_inflated = pd.DataFrame(dtype=str, index=outdeg.index,
                                       columns=outdeg.columns)
for source, target in product(area_list, area_list):
    directionality_inflated.loc[
        (target, slice(None), slice(None)), (source, slice(None), slice(None))
    ] = directionality.loc[target, source]
outdeg_FF = outdeg[directionality_inflated == 'FF']
outdeg_lat = outdeg[directionality_inflated == 'lat']
outdeg_FB = outdeg[directionality_inflated == 'FB']
outdeg_FF = outdeg_FF.groupby('area', axis='index').sum()
outdeg_lat = outdeg_lat.groupby('area', axis='index').sum()
outdeg_FB = outdeg_FB.groupby('area', axis='index').sum()
outdeg_FF_flat = np.ravel(outdeg_FF)
outdeg_FB_flat = np.ravel(outdeg_FB)
outdeg_FF_flat = outdeg_FF_flat[outdeg_FF_flat >= 1]
outdeg_FB_flat = outdeg_FB_flat[outdeg_FB_flat >= 1]
mean_outdeg_FF = np.mean(outdeg_FF_flat)
mean_outdeg_FB = np.mean(outdeg_FB_flat)
print(f'mean outdegree FF: {mean_outdeg_FF}')
print(f'mean outdegree FB: {mean_outdeg_FB}')

# Targets
targets_FF = outdeg_FF[outdeg_FF > target_min_outdeg].fillna(0)
targets_lat = outdeg_lat[outdeg_lat > target_min_outdeg].fillna(0)
targets_FB = outdeg_FB[outdeg_FB > target_min_outdeg].fillna(0)
targets_FF = targets_FF.astype(bool).astype(int).sum(axis='index')
targets_lat = targets_lat.astype(bool).astype(int).sum(axis='index')
targets_FB = targets_FB.astype(bool).astype(int).sum(axis='index')
targets_FF = targets_FF[targets_FF >= 1]
targets_lat = targets_lat[targets_lat >= 1]
targets_FB = targets_FB[targets_FB >= 1]
mean_targets_FF = np.mean(targets_FF)
mean_targets_lat = np.mean(targets_lat)
mean_targets_FB = np.mean(targets_FB)
print(f'mean targets FF: {mean_targets_FF}')
print(f'mean targets lat: {mean_targets_lat}')
print(f'mean targets FB: {mean_targets_FB}')


# ========== Plot ==========
plt.style.use('./misc/mplstyles/report_plots_master.mplstyle')
fig = plt.figure(constrained_layout=True, figsize=(5.63, 3.))
label_prms = dict(fontsize=12, fontweight='bold', va='top', ha='right')
gs = gridspec.GridSpec(
        2,
        2,
        figure=fig
        )

ax_synapses = fig.add_subplot(gs[0, 0])
ax_synapses.spines['top'].set_visible(False)
ax_synapses.spines['right'].set_visible(False)
ax_edr = fig.add_subplot(gs[0, 1])
ax_edr.spines['top'].set_visible(False)
ax_edr.spines['right'].set_visible(False)
ax_outdeg = fig.add_subplot(gs[1, 0])
ax_outdeg.spines['top'].set_visible(False)
ax_outdeg.spines['right'].set_visible(False)
ax_targets = fig.add_subplot(gs[1, 1])
ax_targets.spines['top'].set_visible(False)
ax_targets.spines['right'].set_visible(False)

# Synapses
x_min, x_max = 2, 9
x = np.linspace(x_min, x_max, 1001)
ax_synapses.hist(log_synapses_flat, bins=25, density=True, color='silver')
ax_synapses.plot(x, norm.pdf(x, log_synapses_mean, log_synapses_std), 'k-')
ax_synapses.set_xlim(x_min, x_max)
# ax_synapses.text(s=r'$p_\mathrm{KS}$'+f' > {pval_min}', x=0.7, y=0.8,
#                  transform=ax_synapses.transAxes)
ax_synapses.set_xlabel(r'$\log_{10}$'+'(# synapses)')
ax_synapses.set_ylabel('probability density')
ax_synapses.set_title('Distribution of connection density')
ax_synapses.text(s='A', transform=ax_synapses.transAxes, x=-0.1, y=1.2,
                 **label_prms)

# Distance rule
x_min, x_max = 0, 150
x = np.linspace(x_min, x_max, 1001)
ax_edr.scatter(dist_flat, np.log10(synapses_area_flat), color='silver')
ax_edr.plot(x, edr_intercept+x*edr_slope, color='black')
ax_edr.set_xlim(x_min, x_max)
ax_edr.text(s=r'$\lambda$'+f' = {edr_decay:.1f} mm', x=0.6, y=0.8,
            transform=ax_edr.transAxes)
ax_edr.set_xlabel('Fiber length [mm]')
ax_edr.set_ylabel(r'$\log_{10}$'+'(# synapses)')
ax_edr.set_title('Exponential decay of connection density')
ax_edr.text(s='B', transform=ax_edr.transAxes, x=-0.1, y=1.2, **label_prms)

# Outdegree
bins = np.linspace(0, 7000, 8)
ax_outdeg.hist([outdeg_FF_flat, outdeg_FB_flat], bins=bins, log=True,
               density=True, label=['FF', 'FB'], color=['#8dd3c7', '#bebada'])
ax_outdeg.set_xlim(0, bins[-1])
ax_outdeg.set_xticks(bins)
ax_outdeg.text(s=r'$\mu_\mathrm{FF}$'+f' = {mean_outdeg_FF:.0f}', x=0.55,
               y=0.85, transform=ax_outdeg.transAxes)
ax_outdeg.text(s=r'$\mu_\mathrm{FB}$'+f' = {mean_outdeg_FB:.0f}', x=0.55,
               y=0.75, transform=ax_outdeg.transAxes)
ax_outdeg.set_xlabel('Outdegree')
ax_outdeg.set_ylabel('probability density')
ax_outdeg.set_title('FF/FB difference in outdegree')
ax_outdeg.legend(loc=0)
ax_outdeg.text(s='C', transform=ax_outdeg.transAxes, x=-0.1, y=1.25,
               **label_prms)

# Targts
x_max = max(targets_FF.max(), targets_FB.max(), targets_lat.max())
bins = 2*np.arange(x_max//2+2)
ax_targets.hist([targets_FF, targets_lat, targets_FB], bins=bins, log=True,
                density=True, color=['#8dd3c7', '#ffffb3', '#bebada'],
                stacked=False, label=['FF', 'LAT', 'FB'])
ax_targets.set_xlim(0, bins[-1])
ax_targets.set_xticks(bins)
ax_targets.text(s=r'$\mu_\mathrm{FF}$'+f' = {mean_targets_FF:.2f}', x=0.5,
                y=0.9, transform=ax_targets.transAxes)
ax_targets.text(s=r'$\mu_\mathrm{LAT}$'+f' = {mean_targets_lat:.2f}', x=0.5,
                y=0.8, transform=ax_targets.transAxes)
ax_targets.text(s=r'$\mu_\mathrm{FB}$'+f' = {mean_targets_FB:.2f}', x=0.5,
                y=0.7, transform=ax_targets.transAxes)
ax_targets.set_xlabel('# target areas')
ax_targets.set_ylabel('probability density')
ax_targets.set_title(f'Multiple targets (outdegree > {target_min_outdeg})')
ax_targets.legend(loc=0)
ax_targets.text(s='D', transform=ax_targets.transAxes, x=-0.1, y=1.25,
                **label_prms)

fig.savefig(f'figures/{figname}.pdf')

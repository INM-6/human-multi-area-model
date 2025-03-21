import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde
from scipy.stats import ks_2samp

from ..compute_scaling_experiments.compute_scaling_experiments_helpers import json_load, calc_mean_std
from ..compute_scaling_experiments.compute_scaling_experiments_helpers import right_ordering as fMRI_ordering

experiments = [
        'lower_etoi',
        'lichtman_chii2_smaller_g',
        'lichtman_chii2_distributed_params',
        'lichtman_chii2_different_seed',
        'lichtman_chii2_different_seed_factor10per7'
        ]

if len(sys.argv) > 1 and sys.argv[1] in experiments:
    simulation = sys.argv[1]
else:
    simulation = 'lichtman_chii2_different_seed'

if simulation == 'lichtman_chii2_different_seed':
    from compute_scaling_experiments_helpers import state_scaling_lichtman_chii2_different_seed as state
    save_data_dir = 'simulated_data/scaling_experiment_lichtman_chii2_different_seed'
    name_extension = '_lichtman_chiI2_different_seed'
elif simulation == 'lower_etoi':
    from compute_scaling_experiments_helpers import state_lower_e_to_i as state
    save_data_dir = 'simulated_data/scaling_experiment_lower_e_to_i'
    name_extension = '_lower_e_to_i'
elif simulation == 'lichtman_chii2_smaller_g':
    from compute_scaling_experiments_helpers import state_scaling_lichtman_chii2_smaller_g as state
    save_data_dir = 'simulated_data/scaling_experiment_lichtman_chii2_smaller_g'
    name_extension = '_lichtman_chiI2_smaller_g'
elif simulation == 'lichtman_chii2_distributed_params':
    from compute_scaling_experiments_helpers import state_scaling_lichtman_chii2_distributed_params as state
    save_data_dir = 'simulated_data/scaling_experiment_lichtman_chii2_distributed_params'
    name_extension = '_lichtman_chiI2_distributed_params'
elif simulation == 'lichtman_chii2_different_seed_factor10per7':
    from compute_scaling_experiments_helpers import state_scaling_lichtman_chii2_different_seed_factor10per7 as state
    save_data_dir = 'simulated_data/scaling_experiment_lichtman_chii2_different_seed_factor10per7'
    name_extension = '_lichtman_chiI2_different_seed_factor10per7'
    
linewidth=2.

colors_state = {
        'exp': '#BB5566',  # Experimental
        'groundstate': '#DDAA33',  # Groundstate
        'bestfit': '#004488'  # bestfit state
        }

path_to_chi = json_load(os.path.join(save_data_dir, 'path_to_chi.json'))
for chi, hash_  in path_to_chi.items():
    if hash_ == state['bestfit']:
        bestfit_chi  = float(chi)
        break
    else:
        bestfit_chi = 1.

# ==============================================================================
# Load data for plotting similarity plot
# ==============================================================================

ks_rates_arr = json_load(os.path.join(save_data_dir, 'ks_rates_arr.json'))
ks_cv_arr = json_load(os.path.join(save_data_dir, 'ks_cv_arr.json'))
ks_lvr_arr = json_load(os.path.join(save_data_dir, 'ks_lvr_arr.json'))
df_corr_lh = {}
df_corr_rh = {}
for use_corrcoeff in [True, False]:
    df_corr_lh[use_corrcoeff] = pd.read_csv(
            os.path.join(save_data_dir, f'df_corr_lh_use_corrcoeff_{use_corrcoeff}.csv'),
            header=None,
            names=['chi', 'val'],
            dtype=float
            )
    df_corr_rh[use_corrcoeff] = pd.read_csv(
            os.path.join(save_data_dir, f'df_corr_rh_use_corrcoeff_{use_corrcoeff}.csv'),
            header=None,
            names=['chi', 'val'],
            dtype=float
            )

# ==============================================================================
# Load data for distribution plots
# ==============================================================================

# Experiment
exp_rates = np.load(
        os.path.join(save_data_dir, 'exp_rates.npy'),
        allow_pickle=True
        )
exp_cv = np.load(os.path.join(save_data_dir, 'exp_cv.npy'), allow_pickle=True)
exp_lvr = np.load(os.path.join(save_data_dir, 'exp_lvr.npy'), allow_pickle=True)

# Simulation
sim_rates = {}
sim_lvr = {}
sim_cv = {}

for keys in ['groundstate', 'bestfit']:
    sim_rates[keys] = np.load(
            os.path.join(save_data_dir, f'sim_rates_{keys}.npy'),
            allow_pickle=True
            )
    sim_cv[keys] = np.load(
            os.path.join(save_data_dir, f'sim_cv_{keys}.npy'),
            allow_pickle=True
            )
    sim_lvr[keys] = np.load(
            os.path.join(save_data_dir, f'sim_lvr_{keys}.npy'),
            allow_pickle=True
            )

# ==============================================================================
# Load data for fMRI plot
# ==============================================================================

simulated_fc = {}
for key in ['groundstate', 'bestfit']:
    simulated_fc[key] = pd.read_csv(
            os.path.join(save_data_dir, f'simulated_fc_{key}.csv'),
            index_col=0,
            header=0
            )

# ==============================================================================
# ==============================================================================
# ==============================================================================
# Setup figure
# ==============================================================================
# ==============================================================================
# ==============================================================================

label_prms = dict(fontsize=12, fontweight='bold', va='top', ha='right')

nrows = 3
ncols = 3
width = 5.63
panel_wh_ratio = 1.5

height = width / panel_wh_ratio * float(nrows) / ncols

plt.style.use('misc/mplstyles/report_plots_master.mplstyle')
fig = plt.figure(
        constrained_layout=True,
        figsize=(width, height)
        )

gs = gridspec.GridSpec(
        3,
        1,
        height_ratios=[1,1,2],
        bottom=.1,
        top=.95,
        left=.1,
        right=.95,
        hspace=0.5,
        wspace=1
        )
gs0 = gridspec.GridSpecFromSubplotSpec(
        1,
        3,
        subplot_spec=gs[0],
        wspace=0.4
        )
gs1 = gridspec.GridSpecFromSubplotSpec(
        1,
        3,
        subplot_spec=gs[1],
        hspace=1.0,
        wspace=0.4
        )
gs2 = gridspec.GridSpecFromSubplotSpec(
        1,
        3,
        subplot_spec=gs[2],
        hspace=1.0,
        wspace=0.4
        )

axes = {}
axes['ks'] = plt.subplot(gs0[0,:])

axes['rates'] = plt.subplot(gs1[0])
axes['cv'] = plt.subplot(gs1[1])
axes['lvr'] = plt.subplot(gs1[2])

axes['exp_bold'] = plt.subplot(gs2[0])
axes['groundstate'] = plt.subplot(gs2[1])
axes['bestfit'] = plt.subplot(gs2[2])

axes['ks'].spines['top'].set_visible(False)
axes['ks'].spines['right'].set_visible(False)

axes['rates'].spines['top'].set_visible(False)
axes['rates'].spines['right'].set_visible(False)
axes['cv'].spines['top'].set_visible(False)
axes['cv'].spines['right'].set_visible(False)
axes['lvr'].spines['top'].set_visible(False)
axes['lvr'].spines['right'].set_visible(False)

# ==============================================================================
# Plot ks panel
# ==============================================================================

df_ks_cv = pd.DataFrame({
    'chi': list(ks_cv_arr.keys()),
    'val': list(ks_cv_arr.values())
    },
    dtype=float
    ).sort_values(by='chi')

axes['ks'].plot(
        df_ks_cv['chi'],
        df_ks_cv['val'],
        label='CV ISI',
        color='#66CCEE',
        marker='x',
        linewidth=linewidth
        )

df_ks_lvr = pd.DataFrame({
    'chi': list(ks_lvr_arr.keys()),
    'val': list(ks_lvr_arr.values())
    },
    dtype=float
    ).sort_values(by='chi')

axes['ks'].plot(
        df_ks_lvr['chi'],
        df_ks_lvr['val'],
        label='LvR',
        color='#228833',
        marker='x',
        linewidth=linewidth
        )

df_ks_rates = pd.DataFrame({
    'chi': list(ks_rates_arr.keys()),
    'val': list(ks_rates_arr.values())
    },
    dtype=float
    ).sort_values(by='chi')

axes['ks'].plot(
        df_ks_rates['chi'],
        df_ks_rates['val'],
        color='#CCBB44',
        label='Rates',
        linewidth=linewidth
        )

axes['ks'].plot(
        df_corr_rh[True]['chi'],
        df_corr_rh[True]['val'],
        color='#EE6677',
        label='fMRI\n(Pearson)',
        linewidth=linewidth
        )

axes['ks'].plot(
        df_corr_rh[False]['chi'],
        df_corr_rh[False]['val'],
        color='#4477AA',
        label='fMRI\n(RMSE)',
        linewidth=linewidth
        )


axes['ks'].vlines(bestfit_chi, 0., 1., linestyles='dashed', color='k')
axes['ks'].set_ylim(0, 1)

axes['ks'].set_xlabel('Cortico-cortical scaling $\chi$')
axes['ks'].set_ylabel('Similarity')

box = axes['ks'].get_position()
axes['ks'].set_position([box.x0, box.y0, box.width * 0.9, box.height])
axes['ks'].legend(loc='center left', bbox_to_anchor=(1., 0.5), frameon=False)

axes['ks'].text(
        s='A',
        transform=axes['ks'].transAxes,
        x=-0.085,
        y=1.15,
        **label_prms
        )

# ==============================================================================
# ==============================================================================
# ==============================================================================
# Plot statistics
# ==============================================================================
# ==============================================================================
# ==============================================================================

number_of_trials = 100
number_of_datapoints = 100
last_data_point_rates = 20
last_data_point_cv = 4
last_data_point_lvr = 4
number_of_neurons_rates = min(
        len(exp_rates),
        len(sim_rates['bestfit']),
        len(sim_rates['groundstate'])
        )
number_of_neurons_cv = min(
        len(exp_cv),
        len(sim_cv['bestfit']),
        len(sim_cv['groundstate'])
        )
number_of_neurons_lvr = min(
        len(exp_lvr),
        len(sim_lvr['bestfit']),
        len(sim_lvr['groundstate'])
        )

# ==============================================================================
# Simulated data
# ==============================================================================

for type_ in ['groundstate', 'bestfit']:
    if type_ == 'groundstate':
        label = 'Base version'
    else:
        label = 'Best-fit version'

    # Rates
    vals, bins = np.histogram(sim_rates[type_], bins=np.arange(0.5,20.5,.5))
    vals = vals / np.sum(vals)
    axes['rates'].plot(
            bins[:-1],
            vals,
            color=colors_state[type_],
            linewidth=linewidth
            )

    # CV
    x = np.linspace(0, last_data_point_cv, number_of_datapoints)
    g = gaussian_kde(sim_cv[type_])
    y = g(x)
    cv_mean = y

    axes['cv'].plot(
            x,
            cv_mean,
            color=colors_state[type_],
            linewidth=linewidth
            )

    axes['cv'].hist(
            sim_cv[type_],
            bins=np.linspace(0, 4, 100),
            color=colors_state[type_],
            alpha=.5,
            density=True
            )

    # LvR
    x = np.linspace(0, last_data_point_lvr, number_of_datapoints)
    g = gaussian_kde(sim_lvr[type_])
    y = g(x)
    lvr_mean = y

    axes['lvr'].plot(
            x,
            lvr_mean,
            color=colors_state[type_],
            label=label,
            linewidth=linewidth
            )

    axes['lvr'].hist(
            sim_lvr[type_],
            bins=np.linspace(0, 4, 100),
            color=colors_state[type_],
            alpha=.5,
            density=True
            )

# ==============================================================================
# experimental data
# ==============================================================================

# Rates
vals, bins = np.histogram(exp_rates, bins=np.arange(0.5,20,.5))
vals = vals / np.sum(vals)
axes['rates'].plot(
        bins[:-1],
        vals,
        color=colors_state['exp'],
        linewidth=linewidth
        )

# CV
x = np.linspace(0, last_data_point_cv, number_of_datapoints)
g = gaussian_kde(exp_cv)
y = g(x)
cv_mean = y

axes['cv'].plot(
        x,
        cv_mean,
        color=colors_state['exp'],
        linewidth=linewidth
        )

axes['cv'].hist(
        exp_cv,
        bins=np.linspace(0, last_data_point_cv, number_of_datapoints),
        color=colors_state['exp'],
        alpha=.5,
        density=True
        )

# LvR
x = np.linspace(0, last_data_point_lvr, number_of_datapoints)
g = gaussian_kde(exp_lvr)
y = g(x)
lvr_mean = y

axes['lvr'].plot(
        x,
        lvr_mean,
        color=colors_state['exp'],
        label='Experiment',
        linewidth=linewidth
        )

axes['lvr'].hist(
        exp_lvr,
        bins=np.linspace(0, last_data_point_lvr, number_of_datapoints),
        color=colors_state['exp'],
        alpha=.5,
        density=True
        )

axes['rates'].set_xlim(0.5, 10)
axes['rates'].set_xlabel('Firing rate (spikes/s)')
axes['rates'].set_ylabel('Density')
axes['rates'].set_xscale('log')
axes['rates'].set_ylim(0.0, 0.5)
axes['rates'].text(
        s='B',
        transform=axes['rates'].transAxes,
        x=-0.3,
        y=1.2,
        **label_prms
        )

axes['cv'].set_xlim(0, 3)
axes['cv'].set_xlabel('CV ISI')
axes['cv'].text(
        s='C',
        transform=axes['cv'].transAxes,
        x=-0.15,
        y=1.2,
        **label_prms
        )

axes['lvr'].set_xlim(0, 3)
axes['lvr'].set_xlabel('LvR')
axes['lvr'].legend(bbox_to_anchor=(.45, 0.3), frameon=False)
# axes['ks'].legend(loc='center left', bbox_to_anchor=(1.03, 0.5), frameon=False)
axes['lvr'].text(
        s='D',
        transform=axes['lvr'].transAxes,
        x=-0.15,
        y=1.2,
        **label_prms
        )

# ==============================================================================
# Plot experimental fMRI
# ==============================================================================

data_dir = os.path.join(os.getcwd(), 'experimental_data/senden/rsData_7T_DKparcel/')
roi = pd.read_csv(os.path.join(data_dir, 'ROIs.txt'), header=None, names=['roi'], dtype=str, squeeze=True)
roi = roi.apply(lambda x: x.split('-')[-1])
roi = roi.drop(range(0, 14)).drop(range(82, 85))
areas = roi.values[:34]
BOLD = np.load(os.path.join(data_dir, 'rsDATA_7T_DKparcel.npy'), allow_pickle=True)
BOLD = BOLD[:, 14:82, :]

no_of_persons = BOLD.shape[2]

# There are 600 timesteps in 1.5 second steps in the data
resolution = 1.5
data_points = 600
timesteps = np.arange(data_points) * resolution
# extracted from https://www.nature.com/articles/s41598-017-03420-6#additional-information
# Left hemisphere

clustering = pd.Series(fMRI_ordering)
ordering = clustering.keys()
tmp_lines = clustering.values
lines_border = np.where(tmp_lines[:-1] != tmp_lines[1:])[0]
lines = lines_border + 1
extended_lines = np.append(lines_border, np.array([len(fMRI_ordering) - 1]))

tmp = np.append(np.array([0]), extended_lines)
points = (tmp[1:] + tmp[:-1] ) * .5 + 1
texts = clustering[extended_lines].values

# Extraction of BOLD series into a dictionary of Dataframes of form
# exp_fc[person][hemisphere]
# exp_fc contains the functional connectivities of 19 subjects
exp_fc = {}

for person in range(no_of_persons):
    lh_person = BOLD[:, 0:34, person]
    rh_person = BOLD[:, 34:68, person]
    lh = pd.DataFrame(lh_person, index=timesteps, columns=areas)
    rh = pd.DataFrame(rh_person, index=timesteps, columns=areas)
    BOLD_rest = {}

    # Correlations of all columns, i.e. areas, with each other
    lh_fc = lh.corr()
    rh_fc = rh.corr()

    BOLD_rest['lh'] = lh_fc
    BOLD_rest['rh'] = rh_fc

    if person == 0:
        exp_fc = {
                'lh': lh_fc,
                'rh': rh_fc
                }
    else:
        exp_fc = {
                'lh': exp_fc['lh'] + lh_fc,
                'rh': exp_fc['rh'] + rh_fc
                }

exp_fc = {
        'lh': exp_fc['lh'] / no_of_persons,
        'rh': exp_fc['rh'] / no_of_persons
        }

im = axes['exp_bold'].pcolormesh(
    exp_fc['rh'].loc[ordering][ordering],
    vmin=-1, 
    vmax=1,
    cmap='RdYlBu_r'
)

cbar_ticks = [-1, 0, 1]
cbar = plt.colorbar(
    im,
    ax=axes['exp_bold'],
    ticks=cbar_ticks
)
cbar.set_label(
    'Correlation',
    rotation=270,
    labelpad=15
)
cbar.remove()

axes['exp_bold'].set_xticks(points)
axes['exp_bold'].set_xticklabels(texts, rotation='vertical')
axes['exp_bold'].set_yticks(points)
axes['exp_bold'].set_yticklabels(texts)
axes['exp_bold'].axis('square')

axes['exp_bold'].invert_yaxis()

axes['exp_bold'].hlines(lines, *axes['exp_bold'].get_xlim(), color='k')
axes['exp_bold'].vlines(lines, *axes['exp_bold'].get_xlim(), color='k')

axes['exp_bold'].text(
        s='E',
        transform=axes['exp_bold'].transAxes,
        x=-0.4,
        y=1.2,
        **label_prms
        )

# =========================================================================
# Plot simulated fc
# =========================================================================

for fmri_state in ['groundstate', 'bestfit']:

    """
    Plots structural and functional connectivities
    """

    clustering = pd.Series(fMRI_ordering)
    ordering = clustering.keys()
    tmp_lines = clustering.values
    lines_border = np.where(tmp_lines[:-1] != tmp_lines[1:])[0]
    lines = lines_border + 1
    extended_lines = np.append(lines_border, np.array([len(fMRI_ordering) - 1]))

    tmp = np.append(np.array([0]), extended_lines)
    points = (tmp[1:] + tmp[:-1]) * .5 + 1
    texts = clustering[extended_lines].values

    # Read functional connectivity values
    df_sim_fc_syn = simulated_fc[fmri_state]

    im = axes[fmri_state].pcolormesh(
        df_sim_fc_syn.loc[ordering][ordering],
        vmin=-1,
        vmax=1,
        cmap='RdYlBu_r'
    )

    cbar_ticks = [-1, 0, 1]
    cbar = plt.colorbar(
        im,
        ax=axes[fmri_state],
        ticks=cbar_ticks,
        shrink=.7
    )
    cbar.set_label(
        'Correlation',
        rotation=90,
        labelpad=1
    )
    if fmri_state == 'groundstate':
        cbar.remove()

    axes[fmri_state].set_xticks(points)
    axes[fmri_state].set_xticklabels(texts, rotation='vertical')
    axes[fmri_state].set_yticks(points)
    axes[fmri_state].set_yticklabels(texts)
    axes[fmri_state].axis('square')

    axes[fmri_state].invert_yaxis()

    axes[fmri_state].hlines(lines, *axes[fmri_state].get_xlim(), color='k')
    axes[fmri_state].vlines(lines, *axes[fmri_state].get_xlim(), color='k')

axes['groundstate'].text(
        s='F',
        transform=axes['groundstate'].transAxes,
        x=-0.17,
        y=1.2,
        **label_prms
        )

axes['bestfit'].text(
        s='G',
        transform=axes['bestfit'].transAxes,
        x=-0.17,
        y=1.2,
        **label_prms
        )

fig.savefig(f'figures/figure_scaling_experiment{name_extension}.pdf')

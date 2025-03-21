import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from .compute_scaling_experiments_helpers import get_cc_scalingEtoE, load_data, get_neuron, cvIsi, LVr
from .compute_scaling_experiments_helpers import calculateFuncionalConnectivityCorrelations
from .compute_scaling_experiments_helpers import lvr_from_isi, json_dump




number_of_simulated_2s_snippets = 6
first_snippet = 1

experiments = [
        'lower_etoi',
        'lichtman_chii2_smaller_g',
        'lichtman_chii2_distributed_params',
        'lichtman_chii2_different_seed',
        'lichtman_chii2_random_seeds',
        'lichtman_chii2_different_seed_factor10per7'
        ]

if len(sys.argv) > 1 and sys.argv[1] in experiments:
    simulation = sys.argv[1]
else:
    simulation = 'lichtman_chii2_different_seed'

base_path = os.path.join(os.getcwd(), 'out/')

if simulation == 'lichtman_chii2_different_seed':
    from compute_scaling_experiments_helpers import state_scaling_lichtman_chii2_different_seed as state
    save_data_dir = 'simulated_data/scaling_experiment_lichtman_chii2_different_seed'
    interesting_paths = [
            # These hashes were run using the latest dependencies (Python 3.9).
            '71325878646a5703fc31876955355747/',
            '498a3266bec748385be4603da399d5c1/',
            '0ea37829bdd5902a5f88c932e50d7c94/',
            '8c49a09f51f44fbb036531ce0719b5ba/',
            'ba0017abba66a9a9402001c9df050d2e/',
            '5b14562d75f35e25c4109488d3865533/',
            '90523c45dfad8e5bacb2eaf4d2196f76/',
            'c2319580b4a38f6d3f42cf3a9acff0b5/',
            '5b3ffad279a0575a2795a4e048eaeef5/',
            'd67da5eb34ad040d6addff7d86a4724e/',
            '990a6f7693bebb2d120f4013fcdd43d8/',
            'ecc1d19ed4afae93d03b66b332ba9b3b/',
            '2d05723eb9b2db44beda01b8785bdbf7/',
            '654ff1cc9642ce91e5c5e2c91ce13f61/',
            '33021ab482b709bd4697a5d9dce9e5f8/',
            '6785c5f5661fadb0e5218c05d36e9a9d/',
            'fbfe498ae2bcc4fa2ce72ab7416d9ddc/',
            '6cc8213fba1b6ff7935d0505491045ea/',
            'aa3e9b1f3ef704504bee98665a215178/',
            ]
    sim_path = '4772f0b020c9f3310f4096a6db758343/'
    ana_path = 'fff793e841649603c59db1822e566c93/'
elif simulation == 'lichtman_chii2_random_seeds':
    from compute_scaling_experiments_helpers import state_scaling_lichtman_chii2_random_seeds as state
    save_data_dir = 'simulated_data/scaling_experiment_lichtman_chii2_rseed2903'
    interesting_paths = [
            '498a3266bec748385be4603da399d5c1/',
            '8c49a09f51f44fbb036531ce0719b5ba/',
            '5b3ffad279a0575a2795a4e048eaeef5/',
            'ecc1d19ed4afae93d03b66b332ba9b3b/',
            '2d05723eb9b2db44beda01b8785bdbf7/',
            '654ff1cc9642ce91e5c5e2c91ce13f61/',
            '33021ab482b709bd4697a5d9dce9e5f8/',
            '6785c5f5661fadb0e5218c05d36e9a9d/',
            'fbfe498ae2bcc4fa2ce72ab7416d9ddc/',
            '6cc8213fba1b6ff7935d0505491045ea/',
            'aa3e9b1f3ef704504bee98665a215178/',
            ]
    sim_path = '4772f0b020c9f3310f4096a6db758343/' # 2903
    ana_path = 'fff793e841649603c59db1822e566c93/'
elif simulation == 'lower_etoi':
    from compute_scaling_experiments_helpers import state_lower_e_to_i as state
    save_data_dir = 'simulated_data/scaling_experiment_lower_e_to_i'
    interesting_paths = [
            '04d270ef4972dcc9e4e6938202b000d7/',
            '3c93418f24dbcf6c2d8a33132bc75ca3/',
            '8fd50bf352d999bd5711b74dce77d21c/',
            'da271ef1ca89153e9ef04ad4789dea87/',
            '2440bbd8642a74be8a32b7ca1fa6975e/',
            '4030a96f4369bd3ea2844d0427d1e4d7/',
            '9a8aa3cb8a2cd12632c2bccfa2d30cf2/',
            '31458bed72cb1ba710239cea45ec1cab/',
            '43f736054d798770511d11d6cea634c7/',
            '9d57f3c059228348251735393f046abf/',
            '3ace71d426d1e6fbd34a5dbec1891b7b/',
            '51753977f483813e58131899e2c2c472/',
            'be5b9b57a154b97777664d015cc16cfc/'
            ]
    sim_path = '4772f0b020c9f3310f4096a6db758343/'
    ana_path = '875fb14936529840c819d2bdc86ff14f/'
elif simulation == 'lichtman_chii2_smaller_g':
    from compute_scaling_experiments_helpers import state_scaling_lichtman_chii2_smaller_g as state
    save_data_dir = 'simulated_data/scaling_experiment_lichtman_chii2_smaller_g'
    interesting_paths = [
            '34ab2aecbed4632dd49e311e523976d5/',
            'd7e63a0b8d4f4743a691a3096a225026/',
            'f358fd7d2844f0a6e3e806352af2fe2f/',
            '4dae448eb9c69ad3f2a71972667f3ee4/',
            'e1f0481943c7dd4937436abc3b0ba8ec/',
            'f6de3f7c02703fa1dde65ef14e61171b/'    
            ]
    sim_path = '4772f0b020c9f3310f4096a6db758343/'
    ana_path = '1931d354c159b2a01cba9c4bd0c9ed46/'
elif simulation == 'lichtman_chii2_distributed_params':
    from compute_scaling_experiments_helpers import state_scaling_lichtman_chii2_distributed_params as state
    save_data_dir = 'simulated_data/scaling_experiment_lichtman_chii2_distributed_params'
    interesting_paths = [
            '105ccb56b38cd990b839889f12953a38/',
            '500b3eac17170085d09407a592ea4075/',
            '7b6fa03426dfaa9855113a32fc1a6535/',
            'c207557ee4ca705ba40627631f44d6f7/',
            '3802e13552bc9cb630c340f638ffeb0a/',
            '64bad60cc7e9702f0d81fa1c9f443398/',
            '919efd18ffa0d7046cdce454f8ef6bca/',
            'ea65cf72fd5a1629c4e55f1c0ab33280/',
            '42cb16647d3907d315bad852408cea33/',
            '70029618054442dffacb000422702166/',
            'b2a35418096820454aed549ab4d7a215/'
            ]
    sim_path = '4772f0b020c9f3310f4096a6db758343/'
    ana_path = '1931d354c159b2a01cba9c4bd0c9ed46/'
elif simulation == 'lichtman_chii2_different_seed_factor10per7':
    from compute_scaling_experiments_helpers import state_scaling_lichtman_chii2_different_seed_factor10per7 as state
    save_data_dir = 'simulated_data/scaling_experiment_lichtman_chii2_different_seed_factor10per7'
    interesting_paths = [
            '16de0d322e4b222e6f4713ba662c41da/',
            '66335d4198126b33771291d5467fbf27/',
            '95c2a43bd22a1f27ce65ff551564b557/',
            'd4731e112855447f384827a1281528c5/',
            '19abb84050e7cfadf9fcdf10d5bb0a34/',
            'd9f98fdf4f25bd770b5a8b3afae31697/',
            '69bb1d367ce878bd9bc7ae7cc46185b9/',
            'a44832b116875b8cc03d1a77b7278d7e/',
            '293fc6e4537603ce6e867f563c05049b/',
            '6bc32c0f64d91ae502f07d91b92da12b/',
            'aa3b81bb3f3795c81c9a5b3454e95494/',
            'dff92fb0f6b0e3aa5718b6b43d340a49/',
            '2f75bb3dd5920776a7a720916cd28f71/',
            '6e68141c0b412bd68ecf23ae798d00fe/',
            'ab01c8287e7f70ccc3d3a43801247165/',
            'e3e1c4a3db89d42249965451312aeb75/',
            '300340dca17d9378bf64236bb25f3ae3/',
            '73717e152e908b8ac0831e06ce8c8ffb/',
            'ac304a4d73d44a7e6cda7676c1ff374a/',
            '3619e6131dbfdb258bba41b919f79541/',
            '75065f05aea926f0e4153664f4f9212c/',
            'bc84db13bd75614bd36a563498c142c9/',
            'e891197ebe3eeebb4e555f72273ab92c/',
            '4e3a98b5e43c004f49deba5fe35023f4/',
            '89bd4af3f1044072c5d4b1e634e4d517/',
            'c114dc8dd3981cebd93568871ec1a53a/',
            '58320586de2820a50f47b93962e70f48/',
            '8aca6e3e2da9e7f78559bd9323850983/',
            'c56b28a915a568e44d505296b2f3f83d/',
            'e958fa00425fa488a97292bf48d930af/',
            'e977991b439cc72a0a77154cc97703cb/',
            '5fd951e40a387f8bc20ac7fbcc4f2d5d/',
            'cce7ba1da76cc43ec1d64c7034a78128/',
            'f24a0ce0a69a0fe8ed15bcca21db420e/',
            '642431ea0f4d4cd7885d71f1b2ca9fdf/',
            '93b288add2929814291c19ee2b445e6f/',
            'ce89dacc4fe37ff7873c0a1576ed79d9/',
            'f8d202967ba7cab69ef473165aaa9f74/'
            ]
    sim_path = '4772f0b020c9f3310f4096a6db758343/'
    ana_path = '1931d354c159b2a01cba9c4bd0c9ed46/'

initial_transient = 2500.
exp_st_duration = 2000.
tmax = 12500.

LvR_min_spikes = 10
CV_min_spikes = 10
min_rate = .5

spike_path = base_path + '{}/' + sim_path + 'spikes.pkl'
popgids_path = base_path + '{}/' + sim_path + 'population_GIDs.pkl'
input_current_path = base_path + '{}/' + sim_path + ana_path + 'input_current.pkl'

area = 'caudalanteriorcingulate'

# =========================================================================
# ==============================================================================
# ==============================================================================
# Read in and process data
# =========================================================================
# ==============================================================================
# ==============================================================================

filename = os.path.join(os.getcwd(), 'experimental_data/rutishauser/spikes/')

data = load_data(filename)
print(f'loaded {len(data)} neurons')
all_ids = range(len(data))

# ==============================================================================
# ==============================================================================
# Read in statistics
# ==============================================================================
# ==============================================================================

# ==============================================================================
# Experimental data
# ==============================================================================

exp_rates = []
exp_lvr = []
exp_cv = []
# Loop over all neurons
for id_ in all_ids:
    neuron = get_neuron(data, id_)
    stim_on = neuron['stim_on']
    # timepoint when the simulation is turned on, we want the spiketrain before
    # this. Thus disregard all spikes that follow this. This values is 2.
    baseline = neuron['baseline_stim_on']
    # Every neuron has several recordings. Loop over all these recordings
    for i, st in enumerate(stim_on):
        st = np.atleast_1d(st)
        st = st[st < baseline]
        exp_rates.append(st.size / baseline)
        if st.size >= LvR_min_spikes:
            isi = np.diff(st)
            if isi.min() > 0:
                exp_lvr.append(lvr_from_isi(isi))
        if st.size >= CV_min_spikes:
            isi = np.diff(st)
            if isi.min() > 0:
                exp_cv.append(np.std(isi) / np.mean(isi))
exp_rates = np.array(exp_rates)
exp_rates = exp_rates[exp_rates >= min_rate]
exp_lvr = np.array(exp_lvr)
exp_cv = np.array(exp_cv)

print('Exp', len(exp_rates))

# ==============================================================================
# Simulated data
# ==============================================================================

sim_rates = {}
sim_lvr = {}
sim_cv = {}

for type_, hash_ in state.items():
    spikes = pd.read_pickle(spike_path.format(hash_))
    popgids = pd.read_pickle(popgids_path.format(hash_))

    sim_rates[type_] = []
    sim_lvr[type_] = []
    sim_cv[type_] = []

    spks = spikes.loc[area].agg(list).agg(sum)

    t_max = 2.
    for j in range(first_snippet, number_of_simulated_2s_snippets):
        i = j - 1
        for st in spks:
            st = st[
                    (st > initial_transient + i * exp_st_duration)\
                            & (st < initial_transient + (i+1) * exp_st_duration)
                            ]
            sim_rates[type_].append(st.size / t_max)
            if st.size >= LvR_min_spikes:
                isi = np.diff(st)
                if isi.min() > 0:
                    sim_lvr[type_].append(lvr_from_isi(isi, tau_r=2.))
            if st.size >= CV_min_spikes:
                isi = np.diff(st)
                if isi.min() > 0:
                    sim_cv[type_].append(np.std(isi) / np.mean(isi))
    sim_rates[type_] = np.array(sim_rates[type_])
    sim_rates[type_] = sim_rates[type_][sim_rates[type_] >= min_rate]
    sim_lvr[type_] = np.array(sim_lvr[type_])
    sim_cv[type_] = np.array(sim_cv[type_])

    print(type_, len(sim_rates[type_]))

# ==============================================================================
# Similarity to spiking data expressed via the kolmogorov smirnov test for area
# ==============================================================================

# Filter experimental rates
exp_rates_ = [rate for rate in exp_rates if rate >= min_rate]
path = [os.path.join(base_path, x) for x in interesting_paths]
path_to_chi = {}
ks_rates_arr = {}
ks_cv_arr = {}
ks_lvr_arr = {}
for d in path:
    p_net = d
    p_sim = os.path.join(d, sim_path)
    p_ana = os.path.join(d, sim_path, ana_path)
    if os.path.isdir(p_net):
        cc_scaling = get_cc_scalingEtoE(p_net)
        spikes = pd.read_pickle(os.path.join(p_sim, 'spikes.pkl')).loc[area]
        spks = []
        sim_rates_list = []
        path_to_chi[cc_scaling] = d.split('/')[-2]
        for j in range(first_snippet, number_of_simulated_2s_snippets):
            i = j - 1
            r = (
                    spikes.apply(
                        lambda sts: np.array([st[(st > initial_transient + i * exp_st_duration) & (st < initial_transient + (i+1) * exp_st_duration)].size for st in sts])
                        ) * 1000. / exp_st_duration
                ).agg(list).agg(sum)
            sim_rates_ = [rate for rate in r if rate >= min_rate]
            sim_rates_list += sim_rates_

            spks_tmp = spikes.apply(
                    lambda sts: np.array([st[(st > initial_transient + i * exp_st_duration) & (st < initial_transient + (i+1) * exp_st_duration)] - i*exp_st_duration for st in sts])
                    ).agg(list).agg(sum)
            spks += spks_tmp

        cv, _ = cvIsi(
                spks,
                CV_min_spikes=CV_min_spikes
                )

        lv, _ = LVr(
                spks,
                2,
                LvR_min_spikes=LvR_min_spikes
                )

        ks_rates_arr[cc_scaling] = 1 - ks_2samp(exp_rates_, sim_rates_list)[0]
        ks_cv_arr[cc_scaling] = 1 - ks_2samp(exp_cv, cv)[0]
        ks_lvr_arr[cc_scaling] = 1 - ks_2samp(exp_lvr, lv)[0]

# ==============================================================================
# ==============================================================================
# Calculate correlation between fMRI and simulated fc
# ==============================================================================
# ==============================================================================

df_corr_lh = {}
df_corr_rh = {}
for use_corrcoeff in [True, False]:
    scale = []
    corr_left = []
    corr_right = []
    path = [os.path.join(base_path, x) for x in interesting_paths]
    for d in path:
        p_net = d
        p_ana = os.path.join(d, sim_path, ana_path)
        if os.path.isdir(p_net):
            chi = get_cc_scalingEtoE(p_net)
            scale.append(chi)
            exp__sim_syn__corr_lh, exp__sim_syn__corr_rh, exp_fc_mean_lh, exp_fc_array_rh = calculateFuncionalConnectivityCorrelations(p_ana, tmin=initial_transient, tmax=tmax, use_corrcoeff=use_corrcoeff, base_path=os.path.join(os.getcwd(), ''))
            corr_left.append(exp__sim_syn__corr_lh)
            corr_right.append(exp__sim_syn__corr_rh)

    df_corr_lh[use_corrcoeff] = pd.Series(corr_left, index=scale).sort_index()
    df_corr_rh[use_corrcoeff] = pd.Series(corr_right, index=scale).sort_index()


# ==============================================================================
# Calculate fMRI correlation
# ==============================================================================

# Read values
simulated_fc = {}
for type_, hash_ in state.items():
    curr_in = pd.read_pickle(input_current_path.format(hash_))
    synaptic_currents = curr_in.T

    # Calculate simulated functional connectivity based on synaptic
    # currents
    df_sim_fc_syn = synaptic_currents[
            (synaptic_currents.index >= initial_transient) & (synaptic_currents.index <= tmax)
            ].corr()
    simulated_fc[type_] = df_sim_fc_syn

# ==============================================================================
# ==============================================================================
# Dump data
# ==============================================================================
# ==============================================================================

json_dump(path_to_chi, os.path.join(save_data_dir, 'path_to_chi.json'))

# ==============================================================================
# Dump data for plotting similarity plot
# ==============================================================================

json_dump(ks_rates_arr, os.path.join(save_data_dir, 'ks_rates_arr.json'))
json_dump(ks_cv_arr, os.path.join(save_data_dir, 'ks_cv_arr.json'))
json_dump(ks_lvr_arr, os.path.join(save_data_dir, 'ks_lvr_arr.json'))
for use_corrcoeff in [True, False]:
    df_corr_lh[use_corrcoeff].to_csv(os.path.join(save_data_dir, f'df_corr_lh_use_corrcoeff_{use_corrcoeff}.csv'))
    df_corr_rh[use_corrcoeff].to_csv(os.path.join(save_data_dir, f'df_corr_rh_use_corrcoeff_{use_corrcoeff}.csv'))

# ==============================================================================
# Dump data for distribution plots
# ==============================================================================

# Experiment
exp_rates.dump(os.path.join(save_data_dir, 'exp_rates.npy'))
exp_cv.dump(os.path.join(save_data_dir, 'exp_cv.npy'))
exp_lvr.dump(os.path.join(save_data_dir, 'exp_lvr.npy'))

# Simulation
for key in ['groundstate', 'bestfit']:
    sim_rates[key].dump(os.path.join(save_data_dir, f'sim_rates_{key}.npy'))
    sim_cv[key].dump(os.path.join(save_data_dir, f'sim_cv_{key}.npy'))
    sim_lvr[key].dump(os.path.join(save_data_dir, f'sim_lvr_{key}.npy'))

# ==============================================================================
# Dump data for fMRI plot
# ==============================================================================

for key in ['groundstate', 'bestfit']:
    simulated_fc[key].to_csv(os.path.join(save_data_dir, f'simulated_fc_{key}.csv'))

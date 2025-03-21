import json
import os
import yaml
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import gaussian_kde


state_lower_e_to_i = {
        'groundstate': '04d270ef4972dcc9e4e6938202b000d7',
        'bestfit': '4030a96f4369bd3ea2844d0427d1e4d7',
        }

state_scaling_lichtman_chii2_smaller_g = {
        'groundstate': '4dae448eb9c69ad3f2a71972667f3ee4',
        'bestfit': 'f358fd7d2844f0a6e3e806352af2fe2f',  #chi=1.75, overall best agreement
        }

state_scaling_lichtman_chii2_distributed_params = {
        'groundstate': 'c207557ee4ca705ba40627631f44d6f7',
        'bestfit': '105ccb56b38cd990b839889f12953a38',
        }

state_scaling_lichtman_chii2_different_seed = {
        'groundstate': '90523c45dfad8e5bacb2eaf4d2196f76',
        'bestfit': '8c49a09f51f44fbb036531ce0719b5ba',
        }

state_scaling_lichtman_chii2_random_seeds = {
        'groundstate': '6785c5f5661fadb0e5218c05d36e9a9d',
        'bestfit': '8c49a09f51f44fbb036531ce0719b5ba',
        }

state_scaling_lichtman_chii2_different_seed_factor10per7 = {
        'groundstate': '4e3a98b5e43c004f49deba5fe35023f4',
        'bestfit': 'bc84db13bd75614bd36a563498c142c9',
        }

left_ordering = {'isthmuscingulate': 'DMN',
  'medialorbitofrontal': 'DMN',
  'posteriorcingulate': 'DMN',
  'precuneus': 'DMN',
  'rostralanteriorcingulate': 'DMN',
  'lateralorbitofrontal': 'DMN',
  'parahippocampal': 'DMN',
  'caudalanteriorcingulate': 'DAN',
  'inferiortemporal': 'DAN',
  'middletemporal': 'DAN',
  'parsopercularis': 'DAN',
  'parsorbitalis': 'DAN',
  'parstriangularis': 'DAN',
  'insula': 'SAN',
  'rostralmiddlefrontal': 'SAN',
  'supramarginal': 'SAN',
  'caudalmiddlefrontal': 'SAN',
  'superiortemporal': 'AUD',
  'cuneus': 'VIS',
  'lateraloccipital': 'VIS',
  'fusiform': 'VIS',
  'lingual': 'VIS',
  'bankssts': 'other',
  'entorhinal': 'other',
  'frontalpole': 'other',
  'inferiorparietal': 'other',
  'superiorfrontal': 'other',
  'paracentral': 'other',
  'pericalcarine': 'other',
  'postcentral': 'other',
  'precentral': 'other',
  'superiorparietal': 'other',
  'temporalpole': 'other',
  'transversetemporal': 'other'}

# Right hemisphere:
right_ordering = {'isthmuscingulate': 'DMN',
  'medialorbitofrontal': 'DMN',
  'posteriorcingulate': 'DMN',
  'precuneus': 'DMN',
  'rostralanteriorcingulate': 'DMN',
  'lateralorbitofrontal': 'DMN',
  'parahippocampal': 'DMN',
  'caudalanteriorcingulate': 'DMN',
  'inferiortemporal': 'DAN',
  'middletemporal': 'DAN',
  'parsopercularis': 'DAN',
  'parsorbitalis': 'DAN',
  'parstriangularis': 'DAN',
  'insula': 'SAN',
  'rostralmiddlefrontal': 'SAN',
  'supramarginal': 'SAN',
  'caudalmiddlefrontal': 'SAN',
  'superiortemporal': 'AUD',
  'cuneus': 'VIS',
  'lateraloccipital': 'VIS',
  'fusiform': 'VIS',
  'lingual': 'VIS',
  'bankssts': 'other',
  'entorhinal': 'other',
  'frontalpole': 'other',
  'inferiorparietal': 'other',
  'superiorfrontal': 'other',
  'paracentral': 'other',
  'pericalcarine': 'other',
  'postcentral': 'other',
  'precentral': 'other',
  'superiorparietal': 'other',
  'temporalpole': 'other',
  'transversetemporal': 'other'}

areas = ['bankssts',
         'caudalanteriorcingulate',  # 1
         'caudalmiddlefrontal',
         'cuneus',
         'entorhinal',
         'frontalpole',
         'fusiform',
         'inferiorparietal',
         'inferiortemporal',
         'insula',
         'isthmuscingulate',
         'lateraloccipital',
         'lateralorbitofrontal',
         'lingual',
         'medialorbitofrontal',
         'middletemporal',
         'paracentral',
         'parahippocampal',
         'parsopercularis',
         'parsorbitalis',
         'parstriangularis',
         'pericalcarine',
         'postcentral',
         'posteriorcingulate',
         'precentral',
         'precuneus',
         'rostralanteriorcingulate',  # 2
         'rostralmiddlefrontal',
         'superiorfrontal',  # 3
         'superiorparietal',
         'superiortemporal',
         'supramarginal',
         'temporalpole',
         'transversetemporal']

def json_load(fp):
    """
    Loads a json file and returns its contents.
    """
    with open(fp, 'r') as f:
        data = json.load(f)
    return data

def calc_mean_std(data, number_of_trials, number_of_datapoints,
        number_of_neurons, last_data_point):
    """
    Draws values from an array number_of_trials times and calculates mean and
    std of all drawings.

    Parameters
    ----------
    data : np.array
        Array containing the data
    number_of_trials
        How often do we draw from data to calculate statistics
    number_of_datapoints
        Resolution of mean and std
    number_of_neurons
        How many neurons should be drawn
    last_data_point
        Which is the last datapoint to be taken into account

    Returns
    -------
    arr_mean : np.array
        Full network dictionary
    arr_std : np.array
    x : np.array
    """
    arr = np.empty((number_of_trials, number_of_datapoints))
    x = np.linspace(0, last_data_point, number_of_datapoints)
    for i, _ in enumerate(range(number_of_trials)):
        tmp = np.random.choice(
                data,
                number_of_neurons,
                replace=False
                )
        g = gaussian_kde(tmp)
        y = g(x)
        arr[i] = y
    arr_mean = np.mean(arr, axis=0)
    arr_std = np.std(arr, axis=0)
    return arr_mean, arr_std, x

def lvr_from_isi(isi, tau_r=.002):
    """
    Calculates the lvr from the isi.

    Parameters
    ----------
    isi : np.array
        Array containing the isi
    tau_r: float
        Refactory time of the neurons

    Returns
    -------
    lvr : Float
        The lvr
    """
    lvr = np.sum(
            (1. - 4 * isi[:-1] * isi[1:] / (isi[:-1] + isi[1:]) ** 2) \
            * (1 + 4 * tau_r / (isi[:-1] + isi[1:])) \
            ) * 3 / (isi.size - 1.)
    return lvr

def json_dump(data, fp):
    """
    Dumps the contens of some data to a json file.
    """
    with open(fp, 'w') as f:
        json.dump(data, f)

def get_cc_scalingEtoE(d, key='cc_scalingEtoE'):
    """
    Searches the networks_params file for the value of the scaling factor chi.
    """
    fn = os.path.join(d, 'network_params.yaml')
    with open(fn, 'r') as net:
        net_params = yaml.load(net, Loader=yaml.Loader)
    scale = net_params['scaling_factors_recurrent'][key]
    return scale


def load_data(folder, file='mfc.mat'):
    """
    Loads Rutishauser data
    """
    filename = os.path.join(folder, file)
    data = loadmat(filename, squeeze_me=True, mat_dtype=False,
                   chars_as_strings=True)
    data = data['data_mfc']
    return data


def get_neuron(data, neuron_id):
    """
    Gets information on neurons from Rutishauser data.
    """
    # area codes from end of README.m
    area_dict = {1: 'left amygdala', 2: 'left dACC', 3: 'left hippocampus',
                 4: 'left preSMA', 5: 'right amygdala', 6: 'right dACC',
                 7: 'right hippocampus', 8: 'right preSMA'}
    data_id = data[neuron_id]
    # data structure described in README.m
    neuron = {
        'area': area_dict[data_id['cellinfo'][2]],
        'response_time': data_id['behavior']['RT'][()],
        'stim_on': data_id['ts']['stim_on'][()],
        'baseline_stim_on': data_id['ts']['baseline_stim_on'][()],
        'reply': data_id['ts']['reply'][()],
        'baseline_reply': data_id['ts']['baseline_reply'][()]
    }
    return neuron


def cvIsi(sts, t_start=None, t_stop=None, CV_min_spikes=10):
    """
    Calculates the cv isi for a subset of neurons for a list of spiketrains.
    The spiketrains contained in the list of spiketrains have to contain at
    least 2 spikes for calculating the interspike interval isi. If this
    condition is not met this function returns 0.

    Parameters
    ----------
    sts : list
        list of spiketrains
    t_start : float
        First timestep to take into account
    t_stop :
        Last timestep to take into account

    Returns
    -------
    cv : float
        cv_isi
    """
    # ensure that there are only spiketrains of len > 1. This is
    # for np.diff which needs to output at least one value in order
    # for np.std and np.mean to return something which is not
    # np.nan. Otherwise return 0.
    if t_start:
        sts = [st[st >= t_start] for st in sts]
    if t_stop:
        sts = [st[st <= t_stop] for st in sts]
    sts = [st for st in sts if len(st) > CV_min_spikes]
    if len(sts) > 0:
        isi = np.array([np.diff(x, 1) for x in sts])
        cv = np.array([np.std(x) / np.mean(x) for x in isi])
        cv_isi = np.mean(cv)
        return cv, cv_isi
    return 0.

def calculate_lvr(isi, t_ref):
    """
    Calculates the local variation lv for a interspike interval distribution.

    Parameters
    ----------
    isi : list
        list of interspike intervals isi of a single neuron

    Returns
    -------
    lv : float
        lv
    """
    # NOTE Elephant and mam use different functions. Elephant uses the normal
    # local variation whereas mam uses the revised local variation. LVr depends
    # on firing rate fluctuations which are caused by the refractory period.
    # This can be compensated for by subtracting the refractoriness constant,
    # t_ref, from the ISIs.
    # Here we take the revised local variation.
    # Multi area model function
    val = np.sum(
            (1. - 4 * isi[:-1] * isi[1:] / (isi[:-1] + isi[1:]) ** 2) \
                    * (1 + 4 * t_ref / (isi[:-1] + isi[1:]))
                    ) * 3 / (isi.size - 1.)
    # Elephant function
    # val = 3. * np.mean(np.power(np.diff(isi) / (isi[:-1] + isi[1:]), 2))
    return val


def LVr(sts, t_ref, t_start=None, t_stop=None, LvR_min_spikes=10):
    """
    Calculates the local variation lv for a list of spiketrains sts. First we
    filter for spiktrains of length > 2 because otherwise the calculation
    fails. In this case we return 0. At the end we divide by the number of
    spiketrains, opposed to dividing by the number of neurons in a population.
    This way we take only neurons that have actually spiked into account.

    Parameters
    ----------
    sts : list
        list of spiketrains
    t_start : float
        First timestep to take into account
    t_stop :
        Last timestep to take into account

    Returns
    -------
    lv : float
        sum of single lvs, needs to normalized (=divided by neuron numbers)
    """
    # ensure that there are only spiketrains of len > 2.
    # So every spiketrain st in sts has len(st) > 1.
    if t_start:
        sts = [st[st >= t_start] for st in sts]
    if t_stop:
        sts = [st[st <= t_stop] for st in sts]
    sts = [st for st in sts if len(st) > LvR_min_spikes]
    if len(sts) > 0:
        isi = np.array([np.diff(x, 1) for x in sts])
        lvr = np.array([calculate_lvr(x, t_ref) for x in isi])
        lvr_isi = np.mean(lvr)
        return lvr, lvr_isi
    return 0.

def calculateFuncionalConnectivityCorrelations(ana_path, tmin=2000, tmax=8000,
        base_path=os.path.join(os.getcwd(), 'out'),
        exclude_diagonal=True, use_corrcoeff=True):
    """
    Calculates the correlation between experimental and simulated functional
    correlation.
    """
    curr_in = pd.read_pickle(os.path.join(ana_path, 'input_current.pkl'))
    # Correlations will be written into this file

    # Read in synaptic currents
    synaptic_currents = curr_in.T

    # Correlate synaptic currents, yields a pandas dataframe of simulated
    # functional connectivity based on synaptic input currents
    # (df_sim_fc_syn)
    df_sim_fc_syn = synaptic_currents[(synaptic_currents.index >= tmin) & (synaptic_currents.index <= tmax)].corr()
    if exclude_diagonal:
        np.fill_diagonal(df_sim_fc_syn.values, np.NaN)

    # Sort
    df_sim_fc_syn = df_sim_fc_syn.sort_index(axis=0).sort_index(axis=1)

    # Derive numpy array containing the correlation values. Handier than
    # the underlying pandas dataframe
    sim_fc_syn = df_sim_fc_syn.values.ravel()
    if exclude_diagonal:
        sim_fc_syn = sim_fc_syn[~np.isnan(sim_fc_syn)]

    # =================================================================
    # ============= READ IN AND EXTRACT EXPERIMENTAL DATA =============
    # =================================================================
    # Set Path to experimental data
    data_dir = os.path.join(
            base_path, 'experimental_data', 'senden', 'rsData_7T_DKparcel'
            )

    # Read in regions of interest.
    roi = pd.read_csv(
            os.path.join(data_dir, 'ROIs.txt'),
            header=None, names=['roi'], dtype=str, squeeze=True
            )
    # The rois are given in this manner: ctx-lh-bankssts
    # The name of the area is the last word after -
    roi = roi.apply(lambda x: x.split('-')[-1])
    # The cortical areas are in the range from 14 to 82
    roi = roi.drop(range(0, 14)).drop(range(82, 85))

    # Read in the bold signal. BOLD.shape = (600, 85, 19)
    # First dimension: timesteps
    # Second dimension: Desikan Killiany areas, left (0:34) and right
    # (34:68) hemisphere
    # Third dimension: Participants
    # orientation discrimination, numerosity
    BOLD = np.load(os.path.join(data_dir, 'rsDATA_7T_DKparcel.npy'))
    BOLD = BOLD[:, 14:82, :]

    # There are 600 timesteps in 1.5 second steps in the data
    resolution = 1.5
    timesteps = np.arange(600) * resolution

    # Extract the number of persons
    no_of_persons = BOLD.shape[2]

    # Extraction of Desikan Killiany area names from left hemisphere,
    # i.e. 0:34, and stripping the first 3 characters indicating the
    # hemisphere
    areas = roi.values[:34]

    # Extraction of BOLD series into a dictionary of Dataframes of form
    # exp_fc[person][hemisphere]
    # exp_fc contains the functional connectivities of 19 subjects
    exp_fc = {}

    # Loop over all persons
    for person in range(no_of_persons):
        # Left hemisphere is 0:34, right hemisphere is 34:68
        lh_person = BOLD[:, 0:34, person]
        rh_person = BOLD[:, 34:68, person]

        # BOLD signal into DataFrames
        lh = pd.DataFrame(lh_person, index=timesteps, columns=areas)
        rh = pd.DataFrame(rh_person, index=timesteps, columns=areas)

        # Correlations of all columns, i.e. areas, with each other
        lh_fc = lh.corr()
        rh_fc = rh.corr()

        # Correlation with itself is trivially 1, set those values to
        # nan
        if exclude_diagonal:
            np.fill_diagonal(lh_fc.values, np.NaN)
            np.fill_diagonal(rh_fc.values, np.NaN)

        # Sort and put into dictionary
        lh_fc = lh_fc.sort_index(axis=0).sort_index(axis=1)
        rh_fc = rh_fc.sort_index(axis=0).sort_index(axis=1)
        exp_fc[person] = {
                'lh': lh_fc,
                'rh': rh_fc
                }

    # =================================================================
    # ======================= CALCULATIONS ============================
    # =================================================================
    # Calculate correlations of experimental functional connectivities
    # Convert all functional connectivities to numpy arrays. Correlate
    # these
    # Calculate correlations of all different experimental fcs with
    # simulated fcs
    exp_fc_array_lh = []
    exp_fc_array_rh = []
    exp_fc__sim_fc_syn__array_lh = []
    exp_fc__sim_fc_syn__array_rh = []
    for i in range(no_of_persons):
        tmp_lh = exp_fc[i]['lh'].values.ravel()
        tmp_rh = exp_fc[i]['rh'].values.ravel()
        if exclude_diagonal:
            tmp_lh = tmp_lh[~np.isnan(tmp_lh)]
            tmp_rh = tmp_rh[~np.isnan(tmp_rh)]
        exp_fc_array_lh.append(tmp_lh)
        exp_fc_array_rh.append(tmp_rh)

        exp_fc__sim_fc_syn__tmp_lh = np.corrcoef(sim_fc_syn, tmp_lh)[0, 1]
        exp_fc__sim_fc_syn__tmp_rh = np.corrcoef(sim_fc_syn, tmp_rh)[0, 1]
        exp_fc__sim_fc_syn__array_lh.append(exp_fc__sim_fc_syn__tmp_lh)
        exp_fc__sim_fc_syn__array_rh.append(exp_fc__sim_fc_syn__tmp_rh)

    exp__exp__corr_lh = np.corrcoef(exp_fc_array_lh)
    exp__exp__corr_rh = np.corrcoef(exp_fc_array_rh)

    # Calculate mean correlation between functional connectivities.
    # This gives us to what extent the functional connectivities of the
    # different subjects correspond to each other
    exp__exp_mean__corr_lh = np.sum(
            np.tril(exp__exp__corr_lh, k=-1)
            ).sum() / np.sum(range(no_of_persons))
    exp__exp_mean__corr_rh = np.sum(
            np.tril(exp__exp__corr_rh, k=-1)
            ).sum() / np.sum(range(no_of_persons))


    # Calculate experimental mean functional connectivity
    exp_fc_mean_lh = np.sum(
            [exp_fc[i]['lh'].values.ravel() for i in range(no_of_persons)],
            axis=0
            ) / no_of_persons
    exp_fc_mean_rh = np.sum(
            [exp_fc[i]['rh'].values.ravel() for i in range(no_of_persons)]
            , axis=0
            ) / no_of_persons

    # Diagonal elements gave nans (correlations with themeselves)
    if exclude_diagonal:
        exp_fc_mean_lh = exp_fc_mean_lh[~np.isnan(exp_fc_mean_lh)]
        exp_fc_mean_rh = exp_fc_mean_rh[~np.isnan(exp_fc_mean_rh)]

    # Correlation experimental functional connectivity with simulated
    # functional connectivity based on synaptic currents
    if use_corrcoeff:
        exp__sim_syn__corr_lh = np.corrcoef(
                [exp_fc_mean_lh, sim_fc_syn]
                )[0,1]
        exp__sim_syn__corr_rh = np.corrcoef(
                [exp_fc_mean_rh, sim_fc_syn]
                )[0,1]
    else:
        rmse_lh = np.sqrt(np.mean((exp_fc_mean_lh-sim_fc_syn)**2))
        rmse_exp_lh = np.sqrt(np.mean((exp_fc_mean_lh)**2))
        exp__sim_syn__corr_lh = np.exp(-rmse_lh/rmse_exp_lh)
        rmse_rh = np.sqrt(np.mean((exp_fc_mean_rh-sim_fc_syn)**2))
        rmse_exp_rh = np.sqrt(np.mean((exp_fc_mean_rh)**2))
        exp__sim_syn__corr_rh = np.exp(-rmse_rh/rmse_exp_rh)
    return exp__sim_syn__corr_lh, exp__sim_syn__corr_rh, exp_fc_mean_lh, exp_fc_array_rh

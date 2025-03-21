"""
Provides default parameters used in the simulation.
"""
import os as _os

params = {}
params['outpath'] = _os.path.join(_os.getcwd(), 'out')

# Surface area of a microcircuit [mm^3]
params['surface_area'] = 1

"""
Parameters for cytoarchitecture preprocessing
"""
params['cytoarchitecture_params'] = {
    # Path to cytoarchitecture data
    'src_path': _os.path.join(
        _os.getcwd(),
        'experimental_data',
        'voneconomokoskinas',
        'StructuralData_VonEconomoKoskinas_addedvalues.xls'
    ),
    # Path to ei ratio data
    'ei_ratio_path': _os.path.join(
        _os.getcwd(),
        'experimental_data',
        'fraction_EI',
        'lichtman.csv'
    ),
    # Source parcellation
    'source': 'VonEconomoKoskinas',
    # Target parcellation
    'target': 'DesikanKilliany',
    # Remove layers with fewer neurons than in layer I
    'remove_smaller_layerI': False,
    # Minimal number of neurons per layer
    'min_neurons_per_layer': 5000
}

"""
Parameters for predictive connectomics
"""
params['predictive_connectomic_params'] = {
    'connectivity': 'HcpDesikanKilliany',
    'con_path': _os.path.join(
        _os.getcwd(),
        'experimental_data',
        'hcp_dti',
        'Connectivity_Distances_HCP_DesikanKilliany.mat'
    ),
    'vol_path': _os.path.join(
        _os.getcwd(),
        'experimental_data',
        'hcp_dti',
        'DKAtlas_VolumeAndNames.mat'
    ),
    # Fraction of recurrent (intra-area) connections.
    # Based on value 0.79 for macaque from Markov et al. 2011 and scaling
    # arguments from Herculano-Houzel et al. 2010 to extrapolate to human.
    'FLN': 0.86,
    # Number of synapses per cubic mm.
    # Value from Cano-Astorga et al. (2021) Cerebral Cortex
    # https://doi.org/10.1093/cercor/bhab120
    'rho_syn': 6.6e8,
    # Relative number of cortico-cortical feedback synapses targeting
    # excitatory neurons. Value from Schmidt et al. 2018
    'Z_i': .93,
    # Determines synaptic target pattern (compare Schmidt et al. 2018).
    'SLN_FF': .65,
    # Determines synaptic target pattern (compare Schmidt et al. 2018).
    'SLN_FB': .35,
    # Spatial connectivity decay parameter.
    # Value in micron from Schmidt el al. 2018
    'lmbda': 160,
    # Fit parameters for SLN fit from neuron densities.
    # Value from Schmidt el al. 2018
    'a0': -0.152, 'a1': -1.534
}

"""
Neuron parameters
"""
params['neuron_model_E'] = 'iaf_psc_exp'
params['neuron_model_I'] = 'iaf_psc_exp'
params['neuron_params_E'] = {
    # Leak potential of the neurons (in mV).
    # See Allen Cells GLIF Parameters.ipynb
    'E_L': -70.0,
    # Threshold potential of the neurons (in mV).
    # See Allen Cells GLIF Parameters.ipynb
    'V_th': -45.0,
    # Membrane potential after a spike (in mV).
    # See Allen Cells GLIF Parameters.ipynb
    'V_reset': -70.0,
    # Membrane capacitance (in pF).
    # See Allen Cells GLIF Parameters.ipynb
    'C_m': 220.0,
    # Membrane time constant (in ms).
    # See Allen Cells GLIF Parameters.ipynb
    # Lowered to account for high-conductance state.
    'tau_m': 10.0,
    # Time constant of postsynaptic excitatory currents (in ms).
    # Value for AMPA receptors from (Fourcaud & Brunel, 2002)
    'tau_syn_ex': 2.0,
    # Time constant of postsynaptic inhibitory currents (in ms).
    # Set as the same value as tau_syn_ex
    'tau_syn_in': 2.0,
    # Refractory period of the neurons after a spike (in ms).
    't_ref': 2.0
}
params['neuron_params_I'] = {
    # Leak potential of the neurons (in mV).
    # See Allen Cells GLIF Parameters.ipynb
    'E_L': -70.0,
    # Threshold potential of the neurons (in mV).
    # See Allen Cells GLIF Parameters.ipynb
    'V_th': -45.0,
    # Membrane potential after a spike (in mV).
    # See Allen Cells GLIF Parameters.ipynb
    'V_reset': -70.0,
    # Membrane capacitance (in pF).
    # See Allen Cells GLIF Parameters.ipynb
    'C_m': 100.0,
    # Membrane time constant (in ms).
    # See Allen Cells GLIF Parameters.ipynb
    # Lowered to account for high-conductance state.
    'tau_m': 10.0,
    # Time constant of postsynaptic excitatory currents (in ms).
    # Value for AMPA receptors from (Fourcaud & Brunel, 2002)
    'tau_syn_ex': 2.0,
    # Time constant of postsynaptic inhibitory currents (in ms).
    # Set as the same value as tau_syn_ex
    'tau_syn_in': 2.0,
    # Refractory period of the neurons after a spike (in ms).
    't_ref': 2.0
}
# Distribution of neuron parameters.
# Default relative sd set as 0, meaning no distribution.
# Relative sd set to match the fitted sigma showed in comments below. See Allen Cells GLIF Parameters.ipynb
params['neuron_param_dist_E'] = {
    'V_th': {'distribution': 'lognormal', 'rel_sd': 0.0},  # 0.21
    'C_m': {'distribution': 'lognormal', 'rel_sd': 0.0},   # 0.22
    'tau_m': {'distribution': 'lognormal', 'rel_sd': 0.0}, # 0.55
}
params['neuron_param_dist_I'] = {
    'V_th': {'distribution': 'lognormal', 'rel_sd': 0.0},  # 0.22
    'C_m': {'distribution': 'lognormal', 'rel_sd': 0.0},   # 0.34
    'tau_m': {'distribution': 'lognormal', 'rel_sd': 0.0}, # 0.43
}

"""
Connection parameters
"""
params['connection_params'] = {
    # Relative inhibitory synaptic strength
    'g': -5.,
    # Synaptic weight (mV) for internal input
    # Value from (Eyal et al., 2018) is 0.3mV,
    # lowered to account for high-conductance state.
    'PSP_e': 0.1,
    # Synaptic weight (mV) for external input
    # Value from (Eyal et al., 2018) is 0.3mV,
    # lowered to account for high-conductance state.
    'PSP_ext': 0.1,
    # Relative standard deviation of synaptic weights
    'PSP_rel': 0.1,
    # Transmission probability for any given synapse
    'p_transmit': 1.0
}
# Scaling factors for the recurrent weights
params['scaling_factors_recurrent'] = {
    # Scale local 4E -> 2/3E by a factor 2 (Potjans & Diesmann, 2014)
    'local_scaling4Eto23E': 2.0,
    # Scale local 5E -> I connections within an area, seems to be a
    # good way to stabilize the activity
    'local_scaling5EtoI': 1.0,
    # Scale local E -> I connections, seems to be good to break synch due to
    # recurrent excitatory loops
    'local_scalingEtoI': 1.0,
    # Scale cortico cortical excitatory on excitatory weights
    'cc_scalingEtoE': 1.0,
    # Scale cortico cortical excitatory on inhibitory weights
    'cc_scalingEtoI': 2.0
}
# Scaling factors for the network numbers
params['N_scaling'] = 1.0 # Scaling of population sizes
params['K_scaling'] = 1.0 # Scaling of indegrees
params['scaling_type'] = 'linear' # Type of scaling factor

# Absolute path to the file holding fullscale rates for scaling
# synaptic weights in the network
params['fullscale_rates'] = None

"""
Delays
"""
params['delay_params'] = {
    # Local dendritic delay for excitatory transmission [ms]
    'delay_e': 1.5,
    # Local dendritic delay for inhibitory transmission [ms]
    'delay_i': 0.75,
    # Relative standard deviation for both local and inter-area delays
    'delay_rel': 0.5,
    # Axonal transmission speed to compute interareal delays [mm/ms]
    'interarea_speed': 3.5,
    # Delay distribution (possible: lognormal_clipped and normal_clipped)
    'distribution': 'lognormal_clipped'
}

"""
Input parameters
"""
params['input_params'] = {
    # Strength of the external input (relative to threshold).
    'eta_ext': 1.1
}
# Scaling factors for the external weights
params['scaling_factors_external'] = {
    # Scale input to 5E, seems good to increase the activity in 5E
    'scaling5E': 1.05,
    # Scale input to 6E, seems good to increase the activity in 6E
    'scaling6E': 1.15
}
# Single spike input
params['single_spike'] = {}

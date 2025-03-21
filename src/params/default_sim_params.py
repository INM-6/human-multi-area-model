'''
Provides default parameters used to instantiate the network.
'''

__all__ = [
    "params",
]

params = {
    # Simulation time (in ms).
    't_sim': 1000.0,
    # Resolution of the simulation (in ms).
    'sim_resolution': 0.1,
    # Masterseed for NEST and NumPy.
    'master_seed': 55,
    # Recording interval of the membrane potential (in ms).
    'rec_V_int': 1.0,
    # If True, data will be overwritten,
    # If False, a NESTError is raised if the files already exist.
    'overwrite_files': True,
    # Print the time progress, this should only be used when the simulation
    # is run on a local machine.
    'print_time': False,
    # Set initial value of the membrane potential.
    'V0_mean': -150.,
    # Set initial standard deviation of the membrane potential.
    'V0_sd': 50.,
    # Connectivity initialization rule. Options:
    # * fixed_total_number: fixes the total numer of synapses
    # * fixed_indegree: fixes indegree of single neurons
    'connection_rule': 'fixed_total_number',
    # minimal cortico-cortical delay in ms (0 corresponds to sim_resolution)
    'delay_cc_min': 0.,
    # Recording devices.
    'rec_dev': ['spike_recorder']
}

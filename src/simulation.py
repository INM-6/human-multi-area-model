# -*- coding: utf-8 -*-

import nest
import numpy as np
import os
from copy import deepcopy

from .helpers.lognormal import mu_sigma_lognorm

from .params.default_sim_params import params as _default_params

__all__ = [
    "Simulation",
    "simulationDictFromDump",
]

# visible for notebook
default_params = deepcopy(_default_params)

class Simulation():
    """
    Handles the setup of the network parameters and
    provides functions to connect the network and devices.

    Parameters
    ----------
    sim_dict : dict
        dictionary containing all parameters specific to the simulation
        such as the directory the data is stored in and the seeds
        (see: default_sim_params.py)
    net_dict : dict
         dictionary containing all parameters specific to the neurons
         and the network (see: network_params.py)
    """

    def __init__(self, sim_dict, net_dict):
        self.sim_dict = sim_dict
        self.net_dict = net_dict

    def set_data_path(self, data_path):
        """
        Sets the path for the output files.

        Parameters
        ----------
        data_path : string
        """
        self.data_path = data_path
        if 'spike_recorder' in self.sim_dict['rec_dev']:
            self.spike_path = os.path.join(self.data_path, 'spikes')
        if 'voltmeter' in self.sim_dict['rec_dev']:
            self.volt_path = os.path.join(self.data_path, 'voltages')

        if nest.Rank() == 0:
            if os.path.isdir(data_path):
                print('data directory already exists')
            else:
                os.mkdir(data_path)
                print('data directory created')
            print('Data will be written to %s' % self.data_path)

            if 'spike_recorder' in self.sim_dict['rec_dev']:
                if not os.path.isdir(self.spike_path):
                    os.mkdir(self.spike_path)
            if 'voltmeter' in self.sim_dict['rec_dev']:
                if not os.path.isdir(self.volt_path):
                    os.mkdir(self.volt_path)

    def setup_nest(self, num_threads):
        """
        Hands parameters to the NEST-kernel.

        Resets the NEST-kernel and passes parameters to it.
        The number of seeds for the NEST-kernel is computed, based on the
        total number of MPI processes and threads of each.

        Parameters
        ----------
        num_threads : int
            Local number of threads (per MPI process).
        """
        nest.ResetKernel()
        master_seed = self.sim_dict['master_seed']
        if nest.Rank() == 0:
            print('Master seed: %i ' % master_seed)
        nest.SetKernelStatus(
            {'local_num_threads': num_threads}
            )
        N_tp = nest.GetKernelStatus(['total_num_virtual_procs'])[0]
        if nest.Rank() == 0:
            print('Number of total processes: %i' % N_tp)

        self.sim_resolution = self.sim_dict['sim_resolution']
        kernel_dict = {
            'resolution': self.sim_resolution,
            'rng_seed': master_seed,
            'overwrite_files': self.sim_dict['overwrite_files'],
            'print_time': self.sim_dict['print_time'],
            }
        nest.SetKernelStatus(kernel_dict)

    def create_populations(self):
        """
        Creates the neuronal populations.

        The neuronal populations are created and the parameters are assigned
        to them. The initial membrane potential of the neurons is drawn from a
        normal distribution.
        """
        # Create cortical populations.
        print('Memory on rank {} before creating populations: {:.2f}MB'.format(
            nest.Rank(), self._getMemoryMB()
        ))
        self.pops = {}
        pop_file_name = os.path.join(self.data_path, 'population_GIDs.dat')
        local_num_threads = nest.GetKernelStatus('local_num_threads')
        with open(pop_file_name, 'w+') as pop_file:
            for pop, nn in self.net_dict['neuron_numbers'].items():
                if nn > 0:
                    if pop[-1] == 'E':
                        neuron_model_pop = self.net_dict['neuron_model_E']
                        neuron_params_pop = self.net_dict['neuron_params_E']
                        nrn_prm_dist_pop = self.net_dict['neuron_param_dist_E']
                    elif pop[-1] == 'I':
                        neuron_model_pop = self.net_dict['neuron_model_I']
                        neuron_params_pop = self.net_dict['neuron_params_I']
                        nrn_prm_dist_pop = self.net_dict['neuron_param_dist_I']
                    else:
                        raise NotImplementedError(
                            "Populations have to be E or I"
                        )
                    population = nest.Create(neuron_model_pop, nn)
                    population.set(neuron_params_pop)

                    # Assign DC input
                    nest.SetStatus(
                        population, {
                            'I_e': self.net_dict['dc_drive'].loc[pop]
                        }
                    )

                    population.set(neuron_params_pop)
                    population.V_m = nest.random.normal(
                        mean=self.sim_dict['V0_mean'],
                        std=self.sim_dict['V0_sd']
                    )
                    # Neuron parameters
                    for prm, dist_dict in nrn_prm_dist_pop.items():
                        if dist_dict['rel_sd'] > 0:
                            param_dist = dist_dict['distribution']
                            if param_dist == 'lognormal':
                                mean_prm = neuron_params_pop[prm]
                                offset_prm = 0.
                                if prm == 'V_th':
                                    mean_prm -= neuron_params_pop['E_L']
                                    offset_prm += neuron_params_pop['E_L']
                                assert mean_prm > 0
                                mu_param, sigma_param = mu_sigma_lognorm(
                                    mean=mean_prm,
                                    rel_sd=dist_dict['rel_sd']
                                )
                                population.set({prm:
                                    offset_prm + nest.random.lognormal(
                                    mean=mu_param,
                                    std=sigma_param
                                )})
                            else:
                                err_msg = "Parameter distribution "
                                err_msg += f"{param_dist} not implemented."
                                raise NotImplementedError(err_msg)

                    self.pops[pop] = population
                    pop_file.write('{};{};{}\n'.format(
                        pop, population[0].get()['global_id'], population[-1].get()['global_id']
                    ))
        print('Memory on rank {} after creating populations: {:.2f}MB'.format(
            nest.Rank(), self._getMemoryMB()
        ))

    def create_devices(self):
        """
        Creates the recording devices.

        Only devices which are given in sim_dict['rec_dev'] are created.
        """
        if 'spike_recorder' in self.sim_dict['rec_dev']:
            recdict = {
                'record_to': 'ascii',
                'label': os.path.join(self.spike_path, 'spike_recorder')
            }
            self.spike_recorder = nest.Create('spike_recorder', params=recdict)
        if 'voltmeter' in self.sim_dict['rec_dev']:
            recdictmem = {
                'record_to': 'ascii',
                'label': os.path.join(self.volt_path, 'voltmeter'),
                'record_from': ['V_m'],
            }
            self.voltmeter = nest.Create('multimeter', params=recdictmem)

        if 'spike_recorder' in self.sim_dict['rec_dev']:
            if nest.Rank() == 0:
                print('Spike detectors created')
        if 'voltmeter' in self.sim_dict['rec_dev']:
            if nest.Rank() == 0:
                print('Voltmeters created')

    def create_poisson(self):
        """
        Creates the Poisson generators.

        If Poissonian input is provided, the Poissonian generators are created
        and the parameters needed are passed to the Poissonian generator.
        """
        if nest.Rank() == 0:
            print('Poisson background input created')
        self.poisson = {}
        for pop, nn in self.net_dict['neuron_numbers'].items():
            if nn > 0.:
                sn_ext = self.net_dict['synapses_external'].loc[pop]
                K_ext = sn_ext / nn
                rate_ext = self.net_dict['rate_ext'].loc[pop] * K_ext
                poiss = nest.Create('poisson_generator')
                poiss.set({'rate': rate_ext})
                self.poisson[pop] = poiss

    def create_single_spike(self):
        """
        Creates the single spike generator.
        """
        if nest.Rank() == 0:
            print('Single spike input created')
        self.single_spike = {}
        for pop, spike_time in self.net_dict['spike_time'].items():
            if spike_time >= 0.:
                spike = nest.Create(
                        'spike_generator',
                        params={'spike_times': [spike_time]}
                        )
                self.single_spike[pop] = spike

    def connect_neurons(self):
        """
        Connects the neuronal populations.
        """
        if nest.Rank() == 0:
            print('Connections are established')
        conn_rule = self.sim_dict['connection_rule']
        for (area_i, layer_i, pop_i), target_pop in self.pops.items():
            for (area_j, layer_j, pop_j), source_pop in self.pops.items():
                synapse_nr = self.net_dict['synapses_internal'].loc[
                    (area_i, layer_i, pop_i),
                    (area_j, layer_j, pop_j)
                ]

                if synapse_nr > 0.:
                    if area_i == area_j:
                        min_delay = self.sim_dict['sim_resolution']
                        if pop_j == 'E':
                            mean_delay = self.net_dict['delay_e']
                            std_delay = self.net_dict['delay_e_sd']
                        else:
                            mean_delay = self.net_dict['delay_i']
                            std_delay = self.net_dict['delay_i_sd']
                    else:
                        min_delay = max(self.sim_dict['delay_cc_min'],
                                        self.sim_dict['sim_resolution'])
                        mean_delay = self.net_dict['delay_cc'].loc[
                            area_i, area_j
                        ]
                        std_delay = self.net_dict['delay_cc_sd'].loc[
                            area_i, area_j
                        ]
                    delay_distr = self.net_dict['delay_distribution']
                    if delay_distr == 'normal_clipped':
                        mu_delay = mean_delay
                        sigma_delay = std_delay
                    elif delay_distr == 'lognormal_clipped':
                        mu_delay, sigma_delay = mu_sigma_lognorm(
                            mean=mean_delay, rel_sd=std_delay/mean_delay
                        )
                    else:
                        err_msg = f"Delay distribution {delay_distr}"
                        err_msg += " not implemented."
                        raise NotImplementedError(err_msg)
                    weight = self.net_dict['weights'].loc[
                        (area_i, layer_i, pop_i),
                        (area_j, layer_j, pop_j)
                    ]
                    w_sd = self.net_dict['weights_sd'].loc[
                        (area_i, layer_i, pop_i),
                        (area_j, layer_j, pop_j)
                    ]
                    if conn_rule == 'fixed_total_number':
                        conn_dict_rec = {
                            'rule': conn_rule, 'N': synapse_nr
                        }
                    elif conn_rule == 'fixed_indegree':
                        neuron_nr = self.net_dict['neuron_numbers'].loc[
                            (area_i, layer_i, pop_i)
                        ]
                        indegree = int(np.round(synapse_nr / neuron_nr))
                        conn_dict_rec = {
                            'rule': conn_rule, 'indegree': indegree
                        }
                    else:
                        print(f'Unknown connection rule {conn_rule}.')
                        raise NotImplementedError()
                    if np.isclose(self.net_dict['p_transmit'], 1):
                        syn_dict = {'synapse_model': 'static_synapse'}
                    else:
                        syn_dict = {'synapse_model': 'bernoulli_synapse',
                                    'p_transmit': self.net_dict['p_transmit']}
                        
                    if delay_distr == 'normal_clipped':
                        syn_dict['delay'] = nest.math.max(nest.random.normal(mean=mu_delay, std=sigma_delay), min_delay)
                    elif delay_distr == 'lognormal_clipped':
                        syn_dict['delay'] = nest.math.max(nest.random.lognormal(mean=mu_delay, std=sigma_delay), min_delay)

                    # Skipping connection with weight == 0.0
                    if weight == 0.0:
                        print('Found weight == 0.0 between {} and {}. Skipping connection.'.format(
                            (area_i, layer_i, pop_i),
                            (area_j, layer_j, pop_j)
                            )
                        )
                        continue
                    
                    if weight < 0:
                        syn_dict['weight'] = nest.math.min(nest.random.normal(mean=weight, std=w_sd), 0.0)
                    else:
                        syn_dict['weight'] = nest.math.max(nest.random.normal(mean=weight, std=w_sd), 0.0)
                    nest.Connect(
                        source_pop, target_pop,
                        conn_spec=conn_dict_rec,
                        syn_spec=syn_dict
                    )
            print(
                'Connected all of area {}, layer {} and population {} '
                'on rank {}. Memory: {:.2f} MB.'.format(
                    area_i, layer_i, pop_i,
                    nest.Rank(), self._getMemoryMB()
                )
            )
        print('Memory on rank {} after creating connections: {:.2f}MB'.format(
            nest.Rank(), self._getMemoryMB()
        ))

    def connect_poisson(self):
        """
        Connects the Poisson generators to the microcircuit.
        """
        if nest.Rank() == 0:
            print('Poisson background input is connected')
        for (area_i, layer_i, pop_i), target_pop in self.pops.items():
            conn_dict_poisson = {'rule': 'all_to_all'}
            weight = self.net_dict['weights_ext'].loc[
                (area_i, layer_i, pop_i)
            ]
            w_sd = self.net_dict['weights_ext_sd'].loc[
                (area_i, layer_i, pop_i)
            ]
            syn_dict_poisson = {
                'synapse_model': 'static_synapse',
                'weight': nest.math.max(nest.random.normal(mean=weight, std=w_sd), 0.0),
                'delay': self.sim_dict['sim_resolution']
            }
            nest.Connect(
                self.poisson[(area_i, layer_i, pop_i)], target_pop,
                conn_spec=conn_dict_poisson,
                syn_spec=syn_dict_poisson
            )

    def connect_single_spike(self):
        """
        Connects a spike generator emitting a single spike.
        """
        if nest.Rank() == 0:
            print('Single spike generator input is connected')
        # Loop over all existing
        for pop, spike in self.single_spike.items():
            # Calculate indegree K for scaling weight
            nn = self.net_dict['neuron_numbers'].loc[pop]
            sn_ext = self.net_dict['synapses_external'].loc[pop]
            K_ext = sn_ext / nn
            # Choose first item in list of target gids
            target_pop = [self.pops[pop][0]]
            weight = self.net_dict['weights_ext'].loc[pop]
            # Scale weight
            weight *= 1e3*K_ext
            syn_dict_single_spike = {
                'synapse_model': 'static_synapse',
                'weight': weight,
                }
            nest.Connect(
                spike,
                target_pop,
                syn_spec=syn_dict_single_spike
            )

    def connect_devices(self):
        """ Connects the recording devices to the microcircuit."""
        if nest.Rank() == 0:
            if ('spike_recorder' in self.sim_dict['rec_dev'] and
                    'voltmeter' not in self.sim_dict['rec_dev']):
                print('Spike detector connected')
            elif ('spike_recorder' not in self.sim_dict['rec_dev'] and
                    'voltmeter' in self.sim_dict['rec_dev']):
                print('Voltmeter connected')
            elif ('spike_recorder' in self.sim_dict['rec_dev'] and
                    'voltmeter' in self.sim_dict['rec_dev']):
                print('Spike detector and voltmeter connected')
            else:
                print('no recording devices connected')
        for (area_i, layer_i, pop_i), target_pop in self.pops.items():
            if 'voltmeter' in self.sim_dict['rec_dev']:
                nest.Connect(self.voltmeter, target_pop)
            if 'spike_recorder' in self.sim_dict['rec_dev']:
                nest.Connect(target_pop, self.spike_recorder)

    def setup(self, data_path, num_threads):
        """ Execute subfunctions of the network.

        This function executes several subfunctions to create neuronal
        populations, devices and inputs, connects the populations with
        each other and with devices and input nodes.

        Parameters
        ----------
        data_path : string
        num_threads : int
        """
        self.set_data_path(data_path)
        self.setup_nest(num_threads)
        self.create_populations()
        self.create_devices()
        self.create_poisson()
        self.create_single_spike()
        self.connect_neurons()
        self.connect_poisson()
        self.connect_single_spike()
        self.connect_devices()

    def simulate(self):
        """ Simulates the model."""
        from datetime import datetime

        print("{} Start simulating".format(datetime.now()))
        nest.Simulate(self.sim_dict['t_sim'])
        print("{} Simulation finished".format(datetime.now()))

    def getHash(self):
        """
        Creates a hash from simulation parameters.

        Returns
        -------
        hash : str
            Hash for the simulation
        """
        from dicthash import dicthash

        hash = dicthash.generate_hash_from_dict(self.sim_dict)
        return hash

    def _getMemoryMB(self):
        """
        Return the currently occupied memory for the job in MB.

        Returns
        -------
        currMem : float
            Curently occupied memory in MB
        """
        try:
            currMem = nest.ll_api.sli_func('memory_thisjob')/1024.
        except AttributeError:
            currMem = nest.sli_func('memory_thisjob')/1024.
        return currMem

    def dump(self, base_folder):
        """
        Exports the full simulation specification. Creates a subdirectory of
        base_folder from the simulation hash where it puts all files.

        Parameters
        ----------
        base_folder : string
            Path to base output folder
        """
        import yaml

        hash = self.getHash()
        out_folder = os.path.join(base_folder, hash)
        try:
            os.mkdir(out_folder)
        except OSError:
            pass

        # output simple data as yaml
        fn = os.path.join(out_folder, 'sim.yaml')
        with open(fn, 'w') as outfile:
            yaml.dump(self.sim_dict, outfile, default_flow_style=False)


def simulationDictFromDump(dump_folder):
    """
    Creates a simulation dict from the files created by Simulation.dump().

    Parameters
    ----------
    dump_folder : string`
        Folder with dumped files

    Returns
    -------
    sim_dict : dict
        Full simulation dictionary
    """
    import os
    import yaml

    # Read sim.yaml
    fn = os.path.join(dump_folder, 'sim.yaml')
    with open(fn, 'r') as sim_file:
        sim_dict = yaml.load(sim_file, Loader=yaml.Loader)
    return sim_dict
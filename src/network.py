# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from copy import deepcopy

from .params.default_net_params import params as _default_params

__all__ = [
    "Network",
    "networkDictFromDump",
]

# visible for notebook
default_params = deepcopy(_default_params)

class Network():
    """
    Gathers and prepares all data that is needed for setting up the NEST
    simulation.

    Parameters
    ----------
    NN : NeuronNumbers
        Cytoarchitecture data
    SN : SynapseNumbers
        Connectivity data
    params : dict
        Dictionary containing parameters, compare default_params.py
    """

    def __init__(self, NN, SN, params):
        self.net = {}
        self.params = params
        self.NN = NN
        self.SN = SN

        self.net['area_list'] = NN.area_list
        self.net['layer_list'] = NN.layer_list
        self.net['population_list'] = NN.population_list

        # ===== Neurons =====
        self.net['neuron_model_E'] = params['neuron_model_E']
        self.net['neuron_model_I'] = params['neuron_model_I']
        self.net['neuron_params_E'] = params['neuron_params_E']
        self.net['neuron_params_I'] = params['neuron_params_I']
        self.net['neuron_param_dist_E'] = params['neuron_param_dist_E']
        self.net['neuron_param_dist_I'] = params['neuron_param_dist_I']
        self.net['neuron_numbers'] = NN.getNeuronNumbers()

        # ===== Connections =====
        self.net['SLN'] = SN.getSLN()
        self.net['synapses_internal'] = SN.getSynapseNumbers()
        self.net['synapses_external'] = SN.getSynapseNumbersExternal()
        rel_sd_psp = params['connection_params']['PSP_rel']
        self.net['weights'] = self.weightMatrixInt(
            params['connection_params']['PSP_e'],
            params['connection_params']['g'],
            self.net['neuron_params_E'],
            self.net['neuron_params_I'],
            self.net['neuron_model_E'],
            self.net['neuron_model_I']
        )
        self.net['weights_ext'] = self.weightMatrixExt(
            params['connection_params']['PSP_ext'],
            self.net['neuron_params_E'],
            self.net['neuron_params_I'],
            self.net['neuron_model_E'],
            self.net['neuron_model_I']
        )
        self.net['p_transmit'] = params['connection_params']['p_transmit']

        # ===== Delays =====
        rel_sd_delay = params['delay_params']['delay_rel']
        interarea_speed = params['delay_params']['interarea_speed']
        self.net['delay_e'] = params['delay_params']['delay_e']
        self.net['delay_e_sd'] = np.abs(self.net['delay_e'] * rel_sd_delay)
        self.net['delay_i'] = params['delay_params']['delay_i']
        self.net['delay_i_sd'] = np.abs(self.net['delay_i'] * rel_sd_delay)
        self.net['delay_cc'] = SN.getDistance() / interarea_speed
        self.net['delay_cc_sd'] = np.abs(self.net['delay_cc'] * rel_sd_delay)
        self.net['delay_distribution'] = params['delay_params']['distribution']

        # ===== Input =====
        self.net['eta_ext'] = params['input_params']['eta_ext']
        self.net['rate_ext'] = self.externalRates(self.net['eta_ext'])
        self.net['spike_time'] = self.distribute_singe_spike_times()
        self.net['dc_drive'] = self.add_DC_drive()
        
        # ===== Scale Weights =====
        # has to be done after determining the external rates
        self.net['weights'] = self.scaleWeightsInt(
            **params['scaling_factors_recurrent']
        )
        self.net['weights_sd'] = np.abs(self.net['weights']*rel_sd_psp)
        self.net['weights_ext'] = self.scaleWeightsExt(
            **params['scaling_factors_external']
        )
        self.net['weights_ext_sd'] = np.abs(self.net['weights_ext']*rel_sd_psp)

        # ===== Scale Network =====
        # down-scale the network for testing on low compute resources
        if params['N_scaling'] != 1.0 or params['K_scaling'] != 1.0:
            if params['fullscale_rates'] is None:
                raise ValueError(
                    "fullscale_rates must be provided for scaling the network"
                )
            self.net['fullscale_rates'] = params['fullscale_rates']
            self.net['scaling_type'] = params['scaling_type']
            self.scaleNetwork()

        # ===== Convenience attributes =====
        self.net['NOS'] = SN.NOS
        self.net['SLN'] = SN.SLN
        self.net['directionality'] = SN.directionality

        # ===== Sort all indices for convenience =====
        self.sortIndices()

    def __repr__(self):
        return ('Multi-area network\n'
                'Areas:\n{0}\n'
                'Layers:\n{1}\n'
                'Populations:\n{2}\n'
                'Neurons: {3:.2E}\n'
                'Internal synapses: {4:.2E}\n'
                'External synapses: {5:.2E}\n'
                .format(self.net['area_list'],
                        self.net['layer_list'],
                        self.net['population_list'],
                        self.net['neuron_numbers'].values.sum(),
                        self.net['synapses_internal'].values.sum(),
                        self.net['synapses_external'].values.sum())
                )

    def distribute_singe_spike_times(self):
        """
        Returns a pandas Series containing singular spikes to be emitted to the
        network.

        Returns
        -------
        spike_time : pd.Series
        """
        spike_time = pd.Series(
                data=None,
                index=self.net['neuron_numbers'].index,
                dtype=np.float64
                )
        for target_pop, timepoint in self.params['single_spike'].items():
            assert type(timepoint) == float
            spike_time.loc[target_pop] = timepoint
        return spike_time

    def getNetwork(self):
        """
        Returns fully specified network.

        Returns
        -------
        net : dict
        """
        return self.net

    def weightMatrixInt(self, psp_exc, rel_inh_strength, neuron_params_E,
                        neuron_params_I, neuron_model_E, neuron_model_I):
        """
        Creates the weight matrix of internal weights. Converts PSP's
        to PSC's.

        Parameters
        ----------
        psp_exc : float
            Excitatory postsynaptic potential
        rel_inh_strength : float
            Relative inhibitory postsynaptic potential: psp_inh / psp_exc
        neuron_params : dict
            Parameters of the neuron model
        neuron_model : string
            Neuron model

        Returns
        -------
        weights : DataFrame
            Matrix of PSC's in pA
        """
        # Conversion factor PSP -> PSC
        C_m_E = neuron_params_E['C_m']
        tau_m_E = neuron_params_E['tau_m']
        tau_syn_ex_E = neuron_params_E['tau_syn_ex']
        tau_syn_in_E = neuron_params_E['tau_syn_in']
        PSC_ee_over_PSP_ee = self._getPscOverPsp(
            C_m_E, tau_m_E, tau_syn_ex_E, neuron_model_E
        )
        PSC_ei_over_PSP_ei = self._getPscOverPsp(
            C_m_E, tau_m_E, tau_syn_in_E, neuron_model_E
        )
        C_m_I = neuron_params_I['C_m']
        tau_m_I = neuron_params_I['tau_m']
        tau_syn_ex_I = neuron_params_I['tau_syn_ex']
        tau_syn_in_I = neuron_params_I['tau_syn_in']
        PSC_ii_over_PSP_ii = self._getPscOverPsp(
            C_m_I, tau_m_I, tau_syn_in_I, neuron_model_I
        )
        PSC_ie_over_PSP_ie = self._getPscOverPsp(
            C_m_I, tau_m_I, tau_syn_ex_I, neuron_model_I
        )

        multiindex = pd.MultiIndex.from_product(
            [self.net['area_list'],
             self.net['layer_list'],
             self.net['population_list']],
            names=['area', 'layer', 'population']
        )
        weights = pd.DataFrame(
            data=0,
            dtype=np.float64,
            index=multiindex,
            columns=multiindex
        ).sort_index()
        weights.loc[
            (slice(None), slice(None), 'E'),
            (slice(None), slice(None), 'E')
        ] = PSC_ee_over_PSP_ee * psp_exc
        weights.loc[
            (slice(None), slice(None), 'I'),
            (slice(None), slice(None), 'E')
        ] = PSC_ie_over_PSP_ie * psp_exc
        weights.loc[
            (slice(None), slice(None), 'E'),
            (slice(None), slice(None), 'I')
        ] = PSC_ei_over_PSP_ei * rel_inh_strength * psp_exc
        weights.loc[
            (slice(None), slice(None), 'I'),
            (slice(None), slice(None), 'I')
        ] = PSC_ii_over_PSP_ii * rel_inh_strength * psp_exc

        return weights

    def scaleWeightsInt(self, local_scaling4Eto23E, local_scaling5EtoI,
                        local_scalingEtoI, cc_scalingEtoE, cc_scalingEtoI):
        """
        Scales the weight matrix of internal weights.

        Parameters
        ----------
        local_scaling4Eto23E : float
            Scale local weight from 4E to 23E (compare microcircuit)
        local_scaling5EtoI : float
            Scale local weight from 5E to I
        local_scalingEtoI : float
            Scale local weight from E to I
        cc_scalingEtoE : float
            Scale cortico cortical weight from E to E
        cc_scalingEtoI : float
            Scale cortico cortical weight from E to I

        Returns
        -------
        weights : DataFrame
            Matrix of PSC's in pA
        """
        weights = self.net['weights']
        area_list = self.net['area_list']

        # Scale local weights with respective scaling factors
        for area in area_list:
            # local scaling of 4E to 23E
            if not np.isclose(local_scaling4Eto23E, 1):
                weights.loc[
                    (area, 'II/III', 'E'),
                    (area, 'IV', 'E')
                ] *= local_scaling4Eto23E

            # local scaling of 5E to all I
            if not np.isclose(local_scaling5EtoI, 1):
                weights.loc[
                    (area, slice(None), 'I'),
                    (area, 'V', 'E')
                ] = local_scaling5EtoI * weights.loc[
                    (area, slice(None), 'I'),
                    (area, 'V', 'E')
                ].values

            # local scaling of all E to all I
            if not np.isclose(local_scalingEtoI, 1):
                weights.loc[
                    (area, slice(None), 'I'),
                    (area, slice(None), 'E')
                ] = local_scalingEtoI * weights.loc[
                    (area, slice(None), 'I'),
                    (area, slice(None), 'E')
                ].values

        # Scale cortico cortical weights with respective scaling factors
        for source_area in area_list:
            for target_area in area_list:
                if source_area != target_area:
                    # Excitatory onto excitatory
                    if not np.isclose(cc_scalingEtoE, 1):
                        weights.loc[
                            (target_area, slice(None), 'E'),
                            (source_area, slice(None), 'E')
                        ] = cc_scalingEtoE * weights.loc[
                            (target_area, slice(None), 'E'),
                            (source_area, slice(None), 'E')
                        ].values
                    # Excitatory onto inhibitory
                    if not np.isclose(cc_scalingEtoI, 1):
                        weights.loc[
                            (target_area, slice(None), 'I'),
                            (source_area, slice(None), 'E')
                        ] = cc_scalingEtoI * weights.loc[
                            (target_area, slice(None), 'I'),
                            (source_area, slice(None), 'E')
                        ].values

        return weights

    def weightMatrixExt(self, psp_ext, neuron_params_E, neuron_params_I,
                        neuron_model_E, neuron_model_I):
        """
        Creates the weight vector of external weights. Converts PSP's
        to PSC's.

        Parameters
        ----------
        psp_ext : float
            External postsynaptic potential
        neuron_params : dict
            Parameters of the neuron model
        neuron_model : string
            Neuron model

        Returns
        -------
        weights_ext : Series
            Matrix of PSC's in pA
        """
        # Conversion factor PSP -> PSC
        C_m_E = neuron_params_E['C_m']
        tau_m_E = neuron_params_E['tau_m']
        tau_syn_ex_E = neuron_params_E['tau_syn_ex']
        PSC_ee_over_PSP_ee = self._getPscOverPsp(
            C_m_E, tau_m_E, tau_syn_ex_E, neuron_model_E
        )
        C_m_I = neuron_params_I['C_m']
        tau_m_I = neuron_params_I['tau_m']
        tau_syn_ex_I = neuron_params_I['tau_syn_ex']
        PSC_ie_over_PSP_ie = self._getPscOverPsp(
            C_m_I, tau_m_I, tau_syn_ex_I, neuron_model_I
        )

        multiindex = pd.MultiIndex.from_product(
            [self.net['area_list'],
             self.net['layer_list'],
             self.net['population_list']],
            names=['area', 'layer', 'population']
        )

        weights_ext = pd.Series(
            data=0,
            dtype=np.float64,
            index=multiindex
        )
        weights_ext.loc[
            (slice(None), slice(None), 'E')
        ] = PSC_ee_over_PSP_ee * psp_ext
        weights_ext.loc[
            (slice(None), slice(None), 'I')
        ] = PSC_ie_over_PSP_ie * psp_ext

        return weights_ext

    def scaleWeightsExt(self, scaling5E, scaling6E):
        """
        Scales the weight vector of external weights.

        Parameters
        ----------
        scaling5E : float
            Scale weight to 5E
        scaling6E : float
            Scale weight to 6E

        Returns
        -------
        weights_ext : Series
            Matrix of PSC's in pA
        """
        weights_ext = self.net['weights_ext']
        # Scale weights with respective scaling factors
        if not np.isclose(scaling5E, 1):
            weights_ext.loc[
                (slice(None), 'V', 'E')
            ] = scaling5E * weights_ext.loc[
                (slice(None), 'V', 'E')
            ].values
        if not np.isclose(scaling6E, 1):
            weights_ext.loc[
                (slice(None), 'VI', 'E')
            ] = scaling6E * weights_ext.loc[
                (slice(None), 'VI', 'E')
            ].values
        return weights_ext

    def externalRates(self, eta_ext):
        """
        Calculates population specific external rates such that the mean
        input realtive to threshold equals eta_ext for all populations.

        Parameters
        ----------
        eta_ext : float
            Mean external input relative to threshold

        Returns
        -------
        rates : Series
            Vector of external rates in 1/s
        """
        # neuron parameters
        tau_m_E = self.net['neuron_params_E']['tau_m']
        tau_syn_E = self.net['neuron_params_E']['tau_syn_ex']
        C_m_E = self.net['neuron_params_E']['C_m']
        if self.net['neuron_model_E'] == 'iaf_psc_exp':
            V_th_E = self.net['neuron_params_E']['V_th']
        elif self.net['neuron_model_E'] == 'mat2_psc_exp':
            V_th_E = self.net['neuron_params_E']['omega']
        else:
            raise NotImplementedError(
                "Neuron model {} unknown.".format(self.net['neuron_model_E'])
            )
        E_L_E = self.net['neuron_params_E']['E_L']
        tau_m_I = self.net['neuron_params_I']['tau_m']
        tau_syn_I = self.net['neuron_params_I']['tau_syn_ex']
        C_m_I = self.net['neuron_params_I']['C_m']
        if self.net['neuron_model_I'] == 'iaf_psc_exp':
            V_th_I = self.net['neuron_params_I']['V_th']
        elif self.net['neuron_model_I'] == 'mat2_psc_exp':
            V_th_I = self.net['neuron_params_I']['omega']
        else:
            raise NotImplementedError(
                "Neuron model {} unknown.".format(self.net['neuron_model_I'])
            )
        E_L_I = self.net['neuron_params_I']['E_L']
        # connection parameters
        K_ext = self.net['synapses_external'] / self.net['neuron_numbers']
        K_ext_E = K_ext.loc[(slice(None), slice(None), 'E')]
        K_ext_I = K_ext.loc[(slice(None), slice(None), 'I')]
        W_ext = self.net['weights_ext']
        W_ext_E = W_ext.loc[(slice(None), slice(None), 'E')]
        W_ext_I = W_ext.loc[(slice(None), slice(None), 'I')]
        # conversion factors 1/ms -> mV
        conversion_E = tau_m_E * K_ext_E * tau_syn_E * W_ext_E / C_m_E
        conversion_I = tau_m_I * K_ext_I * tau_syn_I * W_ext_I / C_m_I
        # calculate rates (1e-3 to make [tau_m] = s)
        rates = pd.Series(
            data=0,
            dtype=np.float64,
            index=K_ext.index
        )
        rates.loc[
            (slice(None), slice(None), 'E')
        ] = 1e3 * (V_th_E - E_L_E) * eta_ext / conversion_E
        rates.loc[
            (slice(None), slice(None), 'I')
        ] = 1e3 * (V_th_I - E_L_I) * eta_ext / conversion_I
        rates[self.net['neuron_numbers'] < 1] = 0.
        return rates

    def add_DC_drive(self):
        """
        Adds DC drive to the network.

        Returns
        -------
        dc_drive : pd.Series
            Vector of DC drives
        """
        dc_drive = pd.Series(
            data=0,
            index=self.net['neuron_numbers'].index,
            dtype=np.float64
        )

        return dc_drive
   
    def scaleNetwork(self):
        """
        Scales the network by scaling neuron numbers, synaptic weights and
        indegrees. The scaling factors are given by the parameters
        N_scaling and K_scaling. The scaling is done in the following order:
        1. Scale neuron numbers
        2. Add extra DC drive based on scaled network
        3. Scale synaptic weights and indegrees

        Raises
        ------
        ValueError
            If scaling parameters are not positive
        """
        # Check if scaling parameters are positive
        if self.params['N_scaling'] <= 0 or self.params['K_scaling'] <= 0:
            raise ValueError("Scaling parameters must be positive")

        # Scale neuron numbers
        self.net['neuron_numbers'] = np.round(
            self.net['neuron_numbers'] * self.params['N_scaling']
        ).astype(int)

        # Determine the scaling factor type
        if self.net['scaling_type'] == 'sqrt':
            scaling_factor = np.sqrt(self.params['K_scaling'])
        elif self.net['scaling_type'] == 'linear':
            scaling_factor = self.params['K_scaling']
        else:
            raise ValueError("Unknown scaling type")

        # Scale synaptic weights and indegrees
        self.net['synapses_internal'] = np.round(
            self.net['synapses_internal'] * self.params['N_scaling'] * self.params['K_scaling']
        ).astype(int)
        self.net['weights'] /= scaling_factor

        self.net['synapses_external'] = np.round(
            self.net['synapses_external'] * self.params['N_scaling'] * self.params['K_scaling']
        ).astype(int)
        self.net['weights_ext'] /= scaling_factor

        # Scale synaptic weights standard deviations
        self.net['weights_sd'] /= scaling_factor
        self.net['weights_ext_sd'] /= scaling_factor

        # Add extra DC drive based on scaled network
        self.extraDCforScaledNetwork()

    def extraDCforScaledNetwork(self):
        """
        Add extra DC drive to the network to compensate for scaling and 
        preserve the mean and variance of the input.

        This method adjusts the external synaptic weights and calculates 
        the additional direct current (DC) input required to maintain 
        network stability after scaling. The adjustments are based on 
        neuron parameters and synaptic properties.

        Steps:
        1. Calculate the conversion factors from postsynaptic potential (PSP) 
           to postsynaptic current (PSC) for excitatory and inhibitory neurons.
        2. Adjust the external synaptic weights using the calculated conversion factors.
        3. Create a MultiIndex for neuron parameters based on area, layer, and population.
        4. Initialize tau_m (membrane time constant) and C_m (membrane capacitance) 
           Series with zeros.
        5. Assign tau_m and C_m values for excitatory and inhibitory neurons.
        6. Calculate the x1 term, which represents the external input scaled by 
           synaptic weights, neuron parameters, and external rate.
        7. Calculate the additional DC drive required to compensate for the scaling 
           factor (K_scaling) and update the network's dc_drive attribute.

        Attributes:
        - self.net: Dictionary containing network parameters and properties.
        - self.params: Dictionary containing scaling parameters.

        Returns:
        None
        """
        # Collect network parameters, make sure everything is sorted the same way
        N = self.net['neuron_numbers'].sort_index()
        nu_ext = self.net['rate_ext'].sort_index()  # 1/s
        K = self.net['synapses_internal'].sort_index(axis=0).sort_index(axis=1)
        K = K.div(N, axis=0)
        K_ext = self.net['synapses_external'].sort_index()
        K_ext = K_ext.div(N)
        J = self.net['weights'].sort_index(axis=0).sort_index(axis=1)
        J_ext = self.net['weights_ext'].sort_index()

        # remove empty populations
        mask = N > 0
        N = N.loc[mask]
        nu_ext = nu_ext.loc[mask]
        K = K.loc[mask, mask]
        K_ext = K_ext.loc[mask]
        J = J.loc[mask, mask]
        J_ext = J_ext.loc[mask]

        # Conversion factor PSP -> PSC
        tau_m_E = self.net['neuron_params_E']['tau_m']
        C_m_E = self.net['neuron_params_E']['C_m']
        tau_syn_ex_E = self.net['neuron_params_E']['tau_syn_ex']
        neuron_model_E = self.net['neuron_model_E']
        PSP_ee_over_PSC_ee = self._getPspOverPsc(
            C_m_E, tau_m_E, tau_syn_ex_E, neuron_model_E
        )
        PSP_ei_over_PSC_ei = self._getPspOverPsc(
            C_m_E, tau_m_E, tau_syn_ex_E, neuron_model_E
        )

        tau_m_I = self.net['neuron_params_I']['tau_m']
        C_m_I = self.net['neuron_params_I']['C_m']
        tau_syn_ex_I = self.net['neuron_params_I']['tau_syn_ex']
        neuron_model_I = self.net['neuron_model_I']
        PSP_ie_over_PSC_ie = self._getPspOverPsc(
            C_m_I, tau_m_I, tau_syn_ex_I, neuron_model_I
        )
        PSP_ii_over_PSC_ii = self._getPspOverPsc(
            C_m_I, tau_m_I, tau_syn_ex_I, neuron_model_I
        )

        # Calculate external PSP
        J_ext.loc[
            (slice(None), slice(None), 'E')
        ] *= PSP_ee_over_PSC_ee
        J_ext.loc[
            (slice(None), slice(None), 'I')
        ] *= PSP_ie_over_PSC_ie

        # Calculate PSP of internal synapses
        J.loc[
            (slice(None), slice(None), 'E'),
            (slice(None), slice(None), 'E')
        ] *= PSP_ee_over_PSC_ee
        J.loc[
            (slice(None), slice(None), 'I'),
            (slice(None), slice(None), 'E')
        ] *= PSP_ie_over_PSC_ie
        J.loc[
            (slice(None), slice(None), 'E'),
            (slice(None), slice(None), 'I')
        ] *= PSP_ei_over_PSC_ei
        J.loc[
            (slice(None), slice(None), 'I'),
            (slice(None), slice(None), 'I')
        ] *= PSP_ii_over_PSC_ii

        # Initialize tau_m and C_m Series with zeros
        tau_m = pd.Series(data=0., index=N.index)
        C_m = pd.Series(data=0., index=N.index)

        # Assign tau_m and C_m values for excitatory and inhibitory neurons
        tau_m.loc[
            (slice(None), slice(None), 'E')
        ] = tau_m_E
        tau_m.loc[
            (slice(None), slice(None), 'I')
        ] = tau_m_I

        C_m.loc[
            (slice(None), slice(None), 'E')
        ] = C_m_E
        C_m.loc[
            (slice(None), slice(None), 'I')
        ] = C_m_I

        # Calculate x1 and x1_ext terms

        # Add contribution from external population into the DC drive
        x1_ext = 1e-3 * tau_m * J_ext * K_ext * nu_ext

        # Add contribution from other populations into the DC drive
        full_mean_rates = pd.read_pickle(self.net['fullscale_rates'])
        x1 = 1e-3 * tau_m * (J * K) @ full_mean_rates

        # Determine the scaling factor type
        if self.net['scaling_type'] == 'sqrt':
            scaling_factor = np.sqrt(self.params['K_scaling'])
        elif self.net['scaling_type'] == 'linear':
            scaling_factor = self.params['K_scaling']
        else:
            raise ValueError("Unknown scaling type")

        # Calculate dc_drive
        self.net['dc_drive'] = C_m / tau_m * ((1. - scaling_factor) * (x1_ext + x1))

    def sortIndices(self):
        """
        Sort indices of all Series and DataFrames in self.net
        """
        for key, val in self.net.items():
            if isinstance(val, pd.Series):
                self.net[key] = val.sort_index()
            elif isinstance(val, pd.DataFrame):
                self.net[key] = val.sort_index(axis=0).sort_index(axis=1)

    def getHash(self):
        """
        Creates a hash from all parameters.

        Returns
        -------
        hash : str
            Hash for the network
        """
        from dicthash import dicthash

        elem_net = self._elementarify_dict(self.net)
        hash_ = dicthash.generate_hash_from_dict(elem_net)
        return hash_

    def dump(self, base_folder):
        """
        Exports the full network. Creates a subdirectory of base_folder
        from the network hash where it puts all files.

        Files are separated into DataFrames (*.df) and a yaml file that
        contains all the rest. The names of the df files correspond to their
        key in the network dictionary.

        Also exports all parameters of Network, NeuronNumbers and
        SynapseNumbers.

        Parameters
        ----------
        base_folder : string
            Path to base output folder
        """
        import os
        import yaml

        hash_ = self.getHash()
        out_folder = os.path.join(base_folder, hash_)
        try:
            os.makedirs(out_folder)
        except OSError:
            pass

        # Specify which types have to be put into pickle files
        complex_types = (pd.Series, pd.DataFrame)
        net_simple, net_complex = self._filter_dict(self.net, complex_types)

        # output simple data as yaml
        net_simple = self._elementarify_dict(net_simple)
        fn = os.path.join(out_folder, 'net.yaml')
        with open(fn, 'w') as outfile:
            yaml.dump(net_simple, outfile, default_flow_style=False)

        # output pandas data as pickle
        for key, val in net_complex.items():
            fn = os.path.join(out_folder, '{}.pkl'.format(key))
            val.to_pickle(fn)

        # output parameters as yaml
        fn = os.path.join(out_folder, 'network_params.yaml')
        with open(fn, 'w') as outfile:
            yaml.dump(self.params, outfile, default_flow_style=False)

        # NeuronNumbers parameters as yaml
        fn = os.path.join(out_folder, 'neuronnumbers_params.yaml')
        with open(fn, 'w') as outfile:
            yaml.dump(self.NN.params, outfile, default_flow_style=False)

        # SynapseNumbers parameters as yaml
        fn = os.path.join(out_folder, 'synapsenumbers_params.yaml')
        with open(fn, 'w') as outfile:
            yaml.dump(self.SN.params, outfile, default_flow_style=False)

    @staticmethod
    def _elementarify_dict(d):
        """
        Cast all numpy or pandas objects inside a dict
        to elementary Python types.

        Parameters
        ----------
        d : dict
            dict to modify

        Returns
        -------
        d : dict
            Modified copy of the dict
        """
        from copy import deepcopy

        r = {}
        for key, val in d.items():
            if isinstance(val, (np.float32, np.float64)):
                r[key] = float(val)
            elif isinstance(val, (np.int32, np.int64)):
                r[key] = int(val)
            elif isinstance(val, np.ndarray):
                r[key] = val.tolist()
            elif isinstance(val, pd.Series):
                s = deepcopy(val)
                # Dummy variable because dicthash cannot hash nan
                s = s.fillna(-1.)
                # flatten multiindex
                s.index = ['_'.join(row).strip() for row in s.index.values]
                r[key] = s.to_dict()
            elif isinstance(val, pd.DataFrame):
                df = deepcopy(val)
                # flatten multiindex
                df.index = [
                    '_'.join(row).strip() for row in df.index.values
                ]
                df.columns = [
                    '_'.join(col).strip() for col in df.columns.values
                ]
                r[key] = df.to_dict()
            else:
                r[key] = val
        return r

    @staticmethod
    def _filter_dict(d, type_filter):
        """
        Filters all values of a dict by type.

        Parameters
        ----------
        d : dict
            dict to filter
        type_filter : iterable
            vector of types to filter for

        Returns
        -------
        f : dict
            Filtered dict
        r : dict
            Filtered out remainder of the dict
        """
        f = {}
        r = {}
        for key, val in d.items():
            if isinstance(val, type_filter):
                r[key] = val
            else:
                f[key] = val
        return f, r

    @staticmethod
    def _getPscOverPsp(C_m, tau_m, tau_syn, neuron_model):
        """
        Calculates conversion factor from PSP's to PSC's for neurons
        with exponential postsynaptic currents.
        Compare Potjans_2014.helpers.get_weight

        Parameters
        ----------
        C_m : float
            Membrane potential in pF
        tau_m : float
            Membrane time constant in ms.
        tau_syn : float
            Synaptic time constant in ms.

        Returns
        -------
        PSC_over_PSP : float
        """
        if neuron_model in ['iaf_psc_exp', 'mat2_psc_exp']:
            eps = tau_syn / tau_m
            PSC_over_PSP = C_m * eps**(-1/(1-eps)) / tau_m
        else:
            raise NotImplementedError(
                "Conversion PSP -> PSC for {} unknown.".format(neuron_model)
            )
        return PSC_over_PSP

    @staticmethod
    def _getPspOverPsc(C_m, tau_m, tau_syn, neuron_model):
        """
        Calculates conversion factor from PSC's to PSP's for neurons
        with exponential postsynaptic currents.

        Parameters
        ----------
        C_m : float
            Membrane potential in pF
        tau_m : float
            Membrane time constant in ms.
        tau_syn : float
            Synaptic time constant in ms.

        Returns
        -------
        PSP_over_PSC : float
        """
        if neuron_model in ['iaf_psc_exp', 'mat2_psc_exp']:
            eps = tau_syn / tau_m
            PSP_over_PSC = tau_m / (C_m * eps**(-1/(1-eps))) 
        else:
            raise NotImplementedError(
                "Conversion PSC -> PSP for {} unknown.".format(neuron_model)
            )
        return PSP_over_PSC 

def networkDictFromDump(dump_folder):
    """
    Creates a network dict from the files created by Network.dump().

    Parameters
    ----------
    dump_folder : string`
        Folder with dumped files

    Returns
    -------
    net_dit : dict
        Full network dictionary
    """
    import os
    import yaml
    import pandas as pd

    # Read net.yaml
    fn = os.path.join(dump_folder, 'net.yaml')
    with open(fn, 'r') as net_file:
        net_dict = yaml.load(net_file, Loader=yaml.Loader)

    # Read output parameters
    fn = os.path.join(dump_folder, 'network_params.yaml')
    with open(fn, 'r') as par_file:
        net_dict['network_params'] = yaml.load(par_file, Loader=yaml.Loader)

    # Read NeuronNumbers parameters
    fn = os.path.join(dump_folder, 'neuronnumbers_params.yaml')
    with open(fn, 'r') as par_file:
        net_dict['neuronnumbers_params'] = yaml.load(par_file, Loader=yaml.Loader)

    # Read SynapseNumbers parameters
    fn = os.path.join(dump_folder, 'synapsenumbers_params.yaml')
    with open(fn, 'r') as par_file:
        net_dict['synapsenumbers_params'] = yaml.load(par_file, Loader=yaml.Loader)

    # Read pandas files
    for file in os.listdir(dump_folder):
        fn, fext = os.path.splitext(file)
        if fext == '.pkl':
            net_dict[fn] = pd.read_pickle(os.path.join(dump_folder, file))
    return net_dict
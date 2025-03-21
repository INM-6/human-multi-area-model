# -*- coding: utf-8 -*-

import os
import random
from datetime import datetime
import math
import numpy as np
import pandas as pd
from copy import deepcopy

import matplotlib.pyplot as plt
import seaborn as sns

from .params.default_ana_params import params as _default_params

__all__ = [
    "Analysis",
    "cvIsi",
    "calculate_lv",
    "LV",
    "calc_rates",
    "correlation",
    "instantaneous_spike_count",
    "strip_binned_spiketrains",
    "shell_presort_all_dat",
    "shell_spiketrainify",
    "split_files",
    "kernel_for_psc",
    "analysisDictFromDump",
]

# visible for notebook
default_params = deepcopy(_default_params)

def _timeit(method):
    import time

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        passed_time = round(te - ts, 3)
        print(f'{method.__name__} took {passed_time} s')
        return result
    return timed


class Analysis():
    """
    Class that provides functionality to analyze simulation results.
    """

    def __init__(self, ana_params, net_dict, sim_dict, sim_folder,
                base_path):
        self.ana_dict = ana_params
        self.net_dict = net_dict
        self.sim_dict = sim_dict

        self.base_path = base_path
        self.sim_folder = sim_folder
        self.ana_folder = os.path.join(sim_folder, self.getHash())
        self.plot_folder = os.path.join(self.ana_folder, 'plots')

        seed = self.ana_dict['seed']
        random.seed(seed)

        if not os.path.isdir(self.ana_folder):
            os.mkdir(self.ana_folder)
        print('Results will be written to %s' % self.plot_folder)
        if not os.path.isdir(self.plot_folder):
            os.mkdir(self.plot_folder)
        print('Plots will be written to %s' % self.plot_folder)

        print('{} Reading popGids'.format(datetime.now().time()))
        self.popGids = self._readPopGids()
        print('{} Reading spikes'.format(datetime.now().time()))
        self.spikes = self._readSpikes()

    @_timeit
    def fullAnalysis(self):
        """
        Execute the full analysis.
        """
        print('{} Calculating rate'.format(datetime.now().time()))
        self.rate = self.meanFiringRate()
        print('{} Calculating population CV ISI'.format(datetime.now().time()))
        self.pop_cv_isi = self.popCvIsi()
        print('{} Calculating population LV ISI'.format(datetime.now().time()))
        self.pop_lv = self.popLv()
        print('{} Calculating correlation coefficients.'.format(
            datetime.now().time())
            )
        self.pop_cc = self.popCorrCoeff()
        print('{} Calculating rate histogram'.format(datetime.now().time()))
        self.rate_hist, self.rate_hist_areas = self.firingRateHistogram()
        print('{} Calculating synaptic input currents'.format(
            datetime.now().time())
        )
        self.curr_in = self.synapticInputCurrent()

        print('{} Calculating rates of all neuron'.format(
            datetime.now().time())
        )
        self.individual_rates = self.individualFiringRate()
        print('{} Calculating binned spike rates per neuron'.format(
            datetime.now().time())
        )
        self.spikes_per_neuron_population_resolved, self.spikes_per_neuron_area_resolved = self.binned_spikerates_per_neuron()

        print('{} Calculating BOLD'.format(datetime.now().time()))
        self.BOLD = self.computeBOLD()
        print('{} Calculating BOLD connectivity'.format(datetime.now().time()))
        self.BOLD_correlation = self.calculateBOLDConnectivity()
        print('{} Calculating connectivity correlations'.format(datetime.now().time()))
        self.calculateFuncionalConnectivityCorrelations()
        self.calculateRateDistributionSimilarity()

        print('{} Plotting {}'.format(datetime.now().time(), 'Boxplot'))
        self.plotBoxPlot()
        print('{} Plotting {}'.format(datetime.now().time(), 'Cv Isi'))
        self.plotPopCVIsi()
        print('{} Plotting {}'.format(datetime.now().time(), 'Lv Isi'))
        self.plotPopLV()
        print('{} Plotting {}'.format(
            datetime.now().time(), 'Correlation coefficients')
            )
        self.plotCorrCoff()
        print('{} Plotting binned spike rates per neuron'.format(
            datetime.now().time())
        )
        self.plot_all_binned_spike_rates_area()

        print('{} Plotting functional connectivity based on synaptic input currents'.format(
            datetime.now().time())
        )
        self.plot_functional_connectivity(save_fig=True)
        print('{} Plotting {}'.format(
            datetime.now().time(), 'area averaged spike rates')
            )
        self.plotAllFiringRatesSummary()
        print('{} Plotting {}'.format(
            datetime.now().time(), 'Synaptic currents')
            )
        self.plotAllSynapticCurrentsSummary()
        print('{} Plotting {}'.format(
            datetime.now().time(), 'BOLD signal')
            )
        self.plotAllBOLDSignalSummary()
        for area in self.popGids.index.levels[0]:
            print('{} Plotting {}'.format(datetime.now().time(), area))
            self.plotRasterArea(area)
        print('{} Plotting BOLD connectivities'.format(datetime.now().time()))
        self.plotBOLDConnectivity()
        print('{} Plotting {}'.format(datetime.now().time(), 'Raster statistics'))
        self.plot_raster_statistics(save_fig=True)
        plt.close('all')

    @_timeit
    def meanFiringRate(self):
        """
        Calculates the population averaged firing rate.

        Returns
        -------
        rate : Series
        """
        try:
            rate = pd.read_pickle(os.path.join(self.sim_folder, 'rates.pkl'))
        except FileNotFoundError:
            # Start, End and Simulation time
            t_start = self.ana_dict['meanFiringRate']['t_start']  # in ms
            t_stop = self.ana_dict['meanFiringRate']['t_stop']  # in ms
            if t_start is None:
                t_start = 0.
            if t_stop is None:
                t_stop = self.sim_dict['t_sim']
            t_sim_sec = (t_stop - t_start) / 1000.  # Convert ms to s
            # For all populations we count all spikes that have been emitted
            # during the simulation. Then we divide the total number of spikes
            # per population by the number of neurons in this populations and
            # the simulation time.
            rate = self.spikes.apply(
                    lambda sts: sum([st[(st >= t_start) & (st < t_stop)].size for st in sts])
                    ).div(self.popGids['pop_size']) / t_sim_sec
            rate.to_pickle(os.path.join(self.sim_folder, 'rates.pkl'))
        return rate

    @_timeit
    def individualFiringRate(self):
        """
        Calculates the firing rate of individual neurons.

        Returns
        -------
        rate : Series
        """
        try:
            rate = pd.read_pickle(os.path.join(self.sim_folder, 'rates_individual.pkl'))
        except FileNotFoundError:
            # Start, End and Simulation time
            t_start = self.ana_dict['individualFiringRate']['t_start']  # in ms
            t_stop = self.ana_dict['individualFiringRate']['t_stop']  # in ms
            if t_start is None:
                t_start = 0.
            if t_stop is None:
                t_stop = self.sim_dict['t_sim']
            t_sim_sec = (t_stop - t_start) / 1000.  # Convert ms to s
            # For all neurons we count all spikes that have been emitted
            # during the simulation. Then we divide the total number of spikes
            # by the simulation time.
            rate = self.spikes.apply(
                    lambda sts: np.array([st[(st >= t_start) & (st < t_stop)].size for st in sts])
                    ) / t_sim_sec
            rate.to_pickle(os.path.join(self.sim_folder, 'rates_individual.pkl'))
        return rate

    @_timeit
    def binned_spikerates_per_neuron(self):
        """
        Calculates the distribution of firing rates area and population
        resolved for the complete network.

        Returns
        -------
        spikes_per_neuron_population_resolved : Series
        spikes_per_neuron_area_resolved : Series
        """
        try:
            spikes_per_neuron_population_resolved = pd.read_pickle(os.path.join(self.sim_folder, 'spikes_per_neuron_population_resolved.pkl'))
            spikes_per_neuron_area_resolved = pd.read_pickle(os.path.join(self.sim_folder, 'spikes_per_neuron_area_resolved.pkl'))
        except FileNotFoundError:
            # Simulation time in seconds
            bins = self.ana_dict['binned_spikerates_per_neuron']['bins']
            # For all populations we count all spikes that have been emitted
            # during the simulation. Then we divide the total number of spikes
            # per population by the number of neurons in this populations and
            # the simulation time.
            spikes_per_neuron_population_resolved_tmp = self.spikes.apply(
                    lambda tmp: np.histogram(np.array([len(i) for i in tmp]), bins=bins)
                    )
            spikes_per_neuron_area_resolved_tmp = self.spikes.apply(list).groupby('area').agg(sum).apply(
                    lambda tmp: np.histogram(np.array([len(i) for i in tmp]), bins=bins)
                    )

            tmp = {}
            for area, (hist, bin_edges) in spikes_per_neuron_population_resolved_tmp.items():
                tmp[area + tuple(['hist'])] = hist
                tmp[area + tuple(['bin_edges'])] = bin_edges
            spikes_per_neuron_population_resolved = pd.Series(tmp)

            tmp = {}
            for area, (hist, bin_edges) in spikes_per_neuron_area_resolved_tmp.items():
                tmp[(area, 'hist')] = hist
                tmp[(area, 'bin_edges')] = bin_edges
            spikes_per_neuron_area_resolved = pd.Series(tmp)

            spikes_per_neuron_population_resolved.to_pickle(os.path.join(self.sim_folder, 'spikes_per_neuron_population_resolved.pkl'))
            spikes_per_neuron_area_resolved.to_pickle(os.path.join(self.sim_folder, 'spikes_per_neuron_area_resolved.pkl'))
        return spikes_per_neuron_population_resolved, spikes_per_neuron_area_resolved

    @_timeit
    def firingRateHistogram(self):
        """
        Calculates the time-resolved population averaged firing rate.
        Uses np.histogram with a fixed binsize.

        Returns
        -------
        rate_hist : Series of numpy arrays containing time resolved firing
                    rates
        """
        try:
            rate_hist = pd.read_pickle(os.path.join(
                self.ana_folder, 'rate_histogram.pkl'
            ))
            rate_hist_areas = pd.read_pickle(os.path.join(
                self.ana_folder, 'rate_histogram_areas.pkl'
            ))
        except FileNotFoundError:
            rate_hist = self.spikes.apply(
                    calc_rates,
                    args=(self.sim_dict, self.ana_dict,)
                    ).div(self.popGids['pop_size'])

            rate_hist_areas = self.spikes.apply(list).groupby('area').agg(
                    sum
                    ).apply(
                        calc_rates,
                        args=(self.sim_dict, self.ana_dict)
                    ).div(self.popGids.groupby('area').agg(sum)['pop_size'])

            rate_hist.to_pickle(os.path.join(
                self.ana_folder, 'rate_histogram.pkl'
            ))

            rate_hist_areas.to_pickle(os.path.join(
                self.ana_folder, 'rate_histogram_areas.pkl'
            ))
        return rate_hist, rate_hist_areas

    @_timeit
    def plot_instantaneous_firing_rate(self, save_fig=False):
        """
        Plots the instantaneous firing rate over simulated areas using a heatmap.
        
        Parameters
        ----------
        save_fig : bool, optional
            If True, the figure will be saved to the plot folder. Default is False.
        """
        if not hasattr(self, 'rate_hist_areas'):
            self.rate_hist, self.rate_hist_areas = self.firingRateHistogram()
        
        # Convert rate_hist_areas to spikes/s
        rate_hist_areas = self.rate_hist_areas * 1000
        
        # Convert rate_hist_areas to a DataFrame for easier plotting
        rate_hist_areas_df = pd.DataFrame(rate_hist_areas.tolist(), index=rate_hist_areas.index)

        # Plot the heatmap with an orange-yellow color palette
        plt.style.use('default')
        plt.figure(figsize=(12, 5))
        sns.heatmap(rate_hist_areas_df, cmap='YlOrBr', cbar_kws={'label': 'Spikes/s'}, yticklabels=rate_hist_areas_df.index)
        plt.xlabel('Time (ms)')
        plt.ylabel('Area')
        plt.title('Instantaneous firing rate over simulated areas')
        plt.xticks(rotation=0)  # Rotate x-axis labels to make the times horizontal
        plt.xlim(self.sim_dict['t_sim']-500, self.sim_dict['t_sim'])
        
        # Save the plot if save_fig is True
        if save_fig:
            extension = self.ana_dict['extension']
            plt.savefig(os.path.join(self.plot_folder, f'instantaneous_firing_rate.{extension}'))
        plt.show()

    @_timeit
    def plot_average_rate_per_pop(self, save_fig=False):
        """
        Plots the time-averaged firing rate over simulated populations using a heatmap.
        
        Parameters
        ----------
        save_fig : bool, optional
            If True, the figure will be saved to the plot folder. Default is False.
        """

        # Calculate the time-averaged firing rate if it has not been calculated yet
        if not hasattr(self, 'self.rate'):
            self.rate = self.meanFiringRate()
        mean_rates_per_pop = self.rate

        # Pivot the DataFrame to have areas on the x-axis and layer+pop on the y-axis
        mean_rates_df = mean_rates_per_pop.reset_index().pivot(index=['layer', 'pop'], columns='area', values=0)

        # Create a new index combining layer and pop with layer names converted from Roman to Arabic numerals
        roman_to_arabic = {
            'II/III': '2/3',
            'IV': '4',
            'V': '5',
            'VI': '6',
        }
        mean_rates_df.index = mean_rates_df.index.map(lambda x: f"{roman_to_arabic.get(x[0], x[0])} {x[1]}")

        # Create a mask for NaN values
        mask = mean_rates_df.isna()

        # Plot the heatmap with external grid
        plt.style.use('default')
        plt.figure(figsize=(12, 4.5))
        sns.heatmap(mean_rates_df, cmap='YlOrBr', fmt=".2f", mask=mask, cbar_kws={'label': 'Spikes/s'})
        plt.title('Time-averaged firing rate over simulated populations')
        plt.xlabel('Area')
        plt.ylabel('Population')

        # Rotate x-tick and y-tick labels
        plt.yticks(rotation=0)

        # Plot X marks for NaN values
        for i in range(mean_rates_df.shape[0]):
            for j in range(mean_rates_df.shape[1]):
                if mask.iloc[i, j]:
                    plt.text(j + 0.5, i + 0.5, 'X', ha='center', va='center', color='black', fontsize=12, weight='bold')

        plt.tight_layout()
        
        # Save the plot if save_fig is True
        if save_fig:
            extension = self.ana_dict['extension']
            plt.savefig(os.path.join(self.plot_folder, f'average_rate_per_area.{extension}'))

        # Calculate and print the total mean firing rate
        total_mean_rate = mean_rates_per_pop.mean()
        print(f'Total mean firing rate: {total_mean_rate:.2f} spikes/s')        
        plt.show()

    @_timeit
    def synapticInputCurrent(self):
        """
        Calculates the area averaged synaptic input current.
        Compare MAM dynamics paper, page14.

        Returns
        -------
        current : DataFrame (index=index, columns=t)
        """
        from scipy.signal import convolve

        try:
            curr = pd.read_pickle(os.path.join(
                self.ana_folder, 'input_current.pkl'
            ))
        except FileNotFoundError:
            # Get parameters
            if not hasattr(self, 'rate_hist'):
                self.rate_hist, self.rate_hist_areas = self.firingRateHistogram()
            inst_rates = self.rate_hist
            index = inst_rates.index
            NN = self.net_dict['neuron_numbers'].loc[index]
            weights = self.net_dict['weights'].loc[index, index]
            synapses = self.net_dict['synapses_internal'].loc[index, index]
            indeg = synapses.div(NN, axis=0)
            tau_s_ex_E = self.net_dict['neuron_params_E']['tau_syn_ex']
            tau_s_in_E = self.net_dict['neuron_params_E']['tau_syn_in']
            tau_s_ex_I = self.net_dict['neuron_params_I']['tau_syn_ex']
            tau_s_in_I = self.net_dict['neuron_params_I']['tau_syn_in']
            assert (tau_s_ex_E == tau_s_ex_I) and (tau_s_in_E == tau_s_in_I)
            tau_syn_ex = tau_s_ex_E
            tau_syn_in = tau_s_in_E
            dt = self.sim_dict['sim_resolution']
            binsize = self.ana_dict['rate_histogram']['binsize']
            t_start = 0.
            t_stop = self.ana_dict['rate_histogram']['t_stop']
            if t_stop is None:
                t_stop = self.sim_dict['t_sim']

            kernel_syn_in = kernel_for_psc(tau_syn_in, dt)
            kernel_syn_ex = kernel_for_psc(tau_syn_ex, dt)

            t_in = np.arange(t_start, t_stop, binsize)

            # Apply convolution to all firing rates emitted from a particular
            # population. Depending on whether the population is excitatory or
            # inhibitory different kernels with different synaptic time
            # constants are used. The results are the postynaptic currents
            # originating in a particular population.
            tmp = {}
            for (area, layer, population), dat in inst_rates.items():
                if population == 'E':
                    tau_s = tau_syn_ex
                    kernel = kernel_syn_ex
                else:
                    tau_s = tau_syn_in
                    kernel = kernel_syn_in
                tmp[(area, layer, population)] = convolve(tau_s*dat, kernel,
                                                          mode='same')
            pscs = pd.DataFrame(tmp, index=t_in).T

            # Include weights, indegrees and average
            curr = (weights.abs() * indeg).dot(pscs)
            curr = curr.mul(NN, axis=0).groupby(level=0).sum()
            curr = curr.div(NN.groupby(level=0).sum(), axis=0)
            curr.to_pickle(os.path.join(
                self.ana_folder, 'input_current.pkl'
            ))

        return curr

    @_timeit
    def computeBOLD(self):
        # TODO
        # Use rate ( note it's given in spikes/ms, i.e. multiply by 1000)
        # instead of current. This is in agreement with Gustavo Deco and Viktor
        # Jirsa 2012: Ongoing Cortical Activity at Rest: Criticality,
        # Multistability,and Ghost Attractors
        # and Deco 2019: Brain songs framework used for discovering the
        # relevant timescale of the human brain
        # Kringelbach et al. Dynamic coupling of whole-brain neuronal
        # andneurotransmitter systems
        # It seems, at least in deco 2019 and kringelbach 2020, they only used
        # excitatory neurons
        # Take parameters from Stephan 2017
        from scipy.integrate import solve_ivp
        from scipy.interpolate import interp1d
        import joblib


        try:
            BOLDSIGNAL = pd.read_pickle(os.path.join(
                self.ana_folder, 'bold_signal.pkl'
            ))
        except FileNotFoundError:
            if not hasattr(self, 'rate_hist_areas'):
                _, self.rate_hist_areas = self.firingRateHistogram()
            rate_hist_areas = self.rate_hist_areas * 1000.  # convert
                                                            # spikes / [ms] to
                                                            # spikes / [s]

            def calculate_BOLD(rate, area=None):
                # Assuming the rate is binned in [ms]
                t_vals = np.arange(0, len(rate)) / 1000.
                # Interpolate the rate in order to access values which do
                # not lie on the grid. The ode solver might access values
                # which are slightly out of the bounds of the interpolation
                # range, thus extrapolate those values.
                z_val = interp1d(
                        t_vals,
                        rate,
                        bounds_error=False,
                        fill_value='extrapolate'
                        )

                def E(f_in, rho):
                    """
                    fraction of oxygen extracted from the inflowing blood

                    Buxton et al. 1998
                    """
                    return 1 - np.power(1 - rho, 1/f_in)

                def balloon_windkessel(t, w, z):
                    """
                    f_in : inflow from the venouscompartment
                    v : Volume

                    z : neuronal activity

                    TVB implementations:
                    https://github.com/the-virtual-brain/tvb-hpc/blob/master/tvb_hpc/bold.py
                    https://github.com/the-virtual-brain/tvb-root/blob/bb3d3c91fb2ba20273b9a065943a141002b07229/scientific_library/tvb/analyzers/fmri_balloon.py
                    neurolib implementation:
                    https://github.com/neurolib-dev/neurolib/blob/27ea47aa33d83d080952954600380d33b6e38d34/neurolib/models/bold/timeIntegration.py
                    WholeBrain implementation:
                    https://github.com/dagush/WholeBrain/blob/faf3a557410c2f7a5942c0298c7ae332630d3eb5/functions/BOLDHemModel_Friston2003.py
                    https://github.com/dagush/WholeBrain/blob/master/functions/BOLDHemModel_Stephan2008.py
                    https://github.com/dagush/WholeBrain/blob/master/functions/BOLDHemModel_Stephan2007.py
                    Deco lab
                    https://github.com/decolab/cb-neuromod/blob/master/functions/BOLD.m

                    Important papers:
                    * K.J. Friston, L. Harrison, and W. Penny,
                      Dynamic causal modelling, NeuroImage 19 (2003) 1273–1302
                    * Klaas Enno Stephan, Nikolaus Weiskopf, Peter M. Drysdale, Peter A. Robinson, and Karl J. Friston
                      Comparing hemodynamic models with DCM, NeuroImage 38 (2007) 387–401
                   * K.J. Friston, Katrin H. Preller, Chris Mathys, Hayriye Cagnan, Jakob Heinzle, Adeel Razi, Peter Zeidman
                     Dynamic causal modelling revisited, NeuroImage 199 (2019) 730–744
                    """

                    s, f_in, v, q = w
                    eps = 1.  # Friston 2003; eps = 0.5  Friston 2000
                    kappa = .65  # time-constant for signal decay or
                                 # elimination [1/s], Friston 2003,
                                 # kappa = 1 / tau_s, tau_s = 0.8 in Friston
                                 # 2000
                                 # For some reason some other code sources use
                                 # this value reversed, is this a bug on their
                                 # or our side? cf tvb, wholebrain, deco
                    gamma = .41  # time-constant for autoregulatory feedback
                                 # from blood flow [1/s], Friston 2003,
                                 # gamma = 1 / tau_f
                                 # tau_f = 0.4 in Friston 2000
                                 # For some reason some other code sources use
                                 # this value reversed, is this a bug on their
                                 # or our side? cf tvb, wholebrain, deco
                    tau = .98  # Mean transit time [s], time to traverse the
                               # venous compartment
                    alpha = .32  # Grubb's exponent, Grubb et al 1974
                    rho = .34  # oxygen extraction fraction, sometimes called
                               # E0

                    f_out = np.power(v, 1 / alpha)  # Outflow

                    # ========================================================
                    #  For a description of these equations see Friston 2003
                    # ========================================================

                    # Change of vasodilatory signal s, dependent on neuronal
                    # activity z, which is subject to autoregulatory feedback
                    # Vasodilation is the widening of blood vessels
                    dsdt = eps * z(t) - kappa * s - gamma * (f_in - 1)
                    # assumption of dynamical system linking synaptic activity
                    # and rCBF (regional cerbral blood flow) is linear
                    dfdt = s
                    # Rate of change of volume v
                    dvdt = (f_in - f_out) / tau
                    # change in deoxyhemoglobin q, delivery into venous
                    # compartment minus expelled
                    dqdt = (f_in * E(f_in, rho) / rho - f_out * q / v) / tau

                    w_return = [dsdt, dfdt, dvdt, dqdt]
                    return w_return

                s0 = 0.
                f0 = 1.
                v0 = 1.
                q0 = 1.

                w0 = [s0, f0, v0, q0]

                balloon_windkessel_solution = solve_ivp(
                    lambda t, w: balloon_windkessel(t, w, z_val),
                    (t_vals[0], t_vals[-1]),
                    w0
                    )

                _, _, v, q = balloon_windkessel_solution.y
                time = balloon_windkessel_solution.t

                rho = .34  # oxygen extraction fraction, sometimes called E0
                V0 = 0.02  # resting blood volume fraction

                # Buxton 1998
                k1 = 7 * rho
                k2 = 2.
                k3 = 2 * rho - 0.2

                BOLD = V0 * (k1 * (1 - q) + k2 * (1 - q / v) + k3 * (1 - v))
                return area, BOLD, time

            bold = joblib.Parallel(n_jobs=-1)(
                joblib.delayed(calculate_BOLD)(
                    rate,
                    area
                    ) for area, rate in rate_hist_areas.items()
            )

            tmp = {}
            for area, b, t in bold:
                tmp[(area, 'bold')] = b
                tmp[(area, 't')] = t

            BOLDSIGNAL = pd.Series(tmp)
            BOLDSIGNAL.to_pickle(os.path.join(
                self.ana_folder, 'bold_signal.pkl'
            ))
        return BOLDSIGNAL

    def plotBoxPlot(self):
        """
        Generates a boxplot of the rates of the different populations averaged
        over all areas.
        ----------
        """
        if not hasattr(self, 'self.rate'):
            self.rate = self.meanFiringRate()
        extension = self.ana_dict['extension']
        fig = plt.figure(1)
        area = np.unique(self.rate.index.get_level_values(0))
        layer = np.unique(self.rate.index.get_level_values(1))
        pop_type = np.unique(self.rate.index.get_level_values(2))

        multi_index = pd.MultiIndex.from_product([layer, pop_type])
        ind = [''.join(i) for i in multi_index.tolist()]

        rates = pd.DataFrame(data=0.0, index=area, columns=ind)

        for (a, l, p), r in self.rate.items():
            rates.loc[a, l+p] = r
        ax = sns.boxplot(
            data=rates,
            orient="h",
            palette="Set2",
        )
        col = ['blue', 'red']
        for i in range(len(ind)):
            mybox = ax.patches[i]
            mybox.set_facecolor(col[i % 2])
        plt.xlabel('Rate (spikes/s)')
        plt.ylabel('Population')
        fig.savefig(os.path.join(
            self.plot_folder,
            'boxplot.{0}'.format(extension)
        ))
        plt.clf()
        plt.close(fig)

    @_timeit
    def popCorrCoeff(self):
        """
        Compute correlation coefficients for a subsample of neurons for the
        entire network. Subsample is set to 200 for duration reasons. Higher
        subsample size might yield more accurate results but is costly timing
        wise.

        Returns
        -------
        mean : Pandas Dataframe
            Population-averaged correlation coefficients.
        """
        # import joblib

        try:
            cc = pd.read_pickle(os.path.join(
                self.sim_folder, 'cc.pkl'
            ))
        except FileNotFoundError:
            # lv = self.spikes.apply(LV, args=(t_ref, t_start, t_stop))
            cc = self.spikes.apply(correlation, args=(self.ana_dict, self.sim_dict))
            # cc = joblib.Parallel(n_jobs=-1)(
            #     joblib.delayed(correlation)(
            #         sts,
            #         self.ana_dict
            #         ) for sts in self.spikes.values
            # )
            cc = pd.Series(cc, index=self.spikes.index)

            cc.to_pickle(os.path.join(self.sim_folder, 'cc.pkl'))
        return cc

    def plotCorrCoff(self):
        """
        Generates a boxplot of the correlation coefficients of the different
        populations averaged over all areas.
        ----------
        """
        if not hasattr(self, 'self.pop_cc'):
            self.pop_cc = self.popCorrCoeff()
        extension = self.ana_dict['extension']
        fig = plt.figure(1)
        area = np.unique(self.pop_cc.index.get_level_values(0))
        layer = np.unique(self.pop_cc.index.get_level_values(1))
        pop_type = np.unique(self.pop_cc.index.get_level_values(2))

        multi_index = pd.MultiIndex.from_product([layer, pop_type])
        ind = [''.join(i) for i in multi_index.tolist()]

        pop_ccs = pd.DataFrame(data=0.0, index=area, columns=ind)

        for (a, l, p), r in self.pop_cc.items():
            pop_ccs.loc[a, l+p] = r
        ax = sns.boxplot(
            data=pop_ccs,
            orient="h",
            palette="Set2",
        )
        col = ['blue', 'red']
        for i in range(len(ind)):
            mybox = ax.patches[i]
            mybox.set_facecolor(col[i % 2])
        plt.xlabel('Correlation coefficient')
        plt.ylabel('Population')
        fig.savefig(os.path.join(
            self.plot_folder,
            'cc.{0}'.format(extension)
        ))
        plt.clf()
        plt.close(fig)

    @_timeit
    def popLv(self):
        """
        Compute the Lv value of the spikes.
        See Shinomoto et al. 2009 for details.

        Returns
        -------
        mean : Pandas Dataframe
            Population-averaged Lv.
        """
        try:
            lv = pd.read_pickle(os.path.join(
                self.sim_folder, 'lv.pkl'
            ))
        except FileNotFoundError:
            # Question: divide by calculated lvs or all neurons?
            # Answer: in cv and lv divide by the number of neurons that have
            # actually spiked,
            t_start = self.ana_dict['lvr']['t_start']
            t_stop = self.ana_dict['lvr']['t_stop']
            min_spikes = self.ana_dict['lvr']['min_spikes']
            t_ref_E = self.net_dict['neuron_params_E']['t_ref']
            t_ref_I = self.net_dict['neuron_params_I']['t_ref']
            # TODO different refractory periods
            assert(t_ref_E == t_ref_I)
            t_ref = t_ref_E
            lv = self.spikes.apply(LV, args=(t_ref, t_start, t_stop,
                                             min_spikes))
            lv.to_pickle(os.path.join(self.sim_folder, 'lv.pkl'))
        return lv

    def plotPopLV(self):
        """
        Generates a boxplot of the rates of the different populations averaged
        over all areas.
        ----------
        """
        if not hasattr(self, 'self.pop_lv'):
            self.pop_lv = self.popLv()
        extension = self.ana_dict['extension']
        fig = plt.figure(1)
        area = np.unique(self.pop_lv.index.get_level_values(0))
        layer = np.unique(self.pop_lv.index.get_level_values(1))
        pop_type = np.unique(self.pop_lv.index.get_level_values(2))

        multi_index = pd.MultiIndex.from_product([layer, pop_type])
        ind = [''.join(i) for i in multi_index.tolist()]

        pop_lvs = pd.DataFrame(data=0.0, index=area, columns=ind)

        for (a, l, p), r in self.pop_lv.items():
            pop_lvs.loc[a, l+p] = r
        ax = sns.boxplot(
            data=pop_lvs,
            orient="h",
            palette="Set2",
        )
        col = ['blue', 'red']
        for i in range(len(ind)):
            mybox = ax.patches[i]
            mybox.set_facecolor(col[i % 2])
        plt.xlabel('Lv (spikes/s)')
        plt.ylabel('Population')
        fig.savefig(os.path.join(
            self.plot_folder,
            'lv_isi.{0}'.format(extension)
        ))
        plt.clf()
        plt.close(fig)

    @_timeit
    def popCvIsi(self):
        """
        Calculates the coefficient of variation cv of the interspike intervals
        isi.

        Returns
        -------
        mean : Pandas Dataframe
            Mean CV ISI value of the population
        """
        try:
            cv_isi = pd.read_pickle(os.path.join(
                self.sim_folder, 'cv_isi.pkl'
            ))
        except FileNotFoundError:
            # Question: divide by calculated cv isis or all neurons?
            # Answer: in cv and lv divide by the number of neurons that have
            # actually spiked,
            t_start = self.ana_dict['cv']['t_start']
            t_stop = self.ana_dict['cv']['t_stop']
            min_spikes = self.ana_dict['cv']['min_spikes']
            cv_isi = self.spikes.apply(cvIsi, args=(t_start, t_stop,
                                                    min_spikes))
            cv_isi.to_pickle(os.path.join(self.sim_folder, 'cv_isi.pkl'))
        return cv_isi

    def plotPopCVIsi(self):
        """
        Generates a boxplot of the rates of the different populations averaged
        over all areas.
        ----------
        """
        if not hasattr(self, 'self.pop_cv_isi'):
            self.pop_cv_isi = self.popCvIsi()
        extension = self.ana_dict['extension']
        fig = plt.figure(1)
        area = np.unique(self.pop_cv_isi.index.get_level_values(0))
        layer = np.unique(self.pop_cv_isi.index.get_level_values(1))
        pop_type = np.unique(self.pop_cv_isi.index.get_level_values(2))

        multi_index = pd.MultiIndex.from_product([layer, pop_type])
        ind = [''.join(i) for i in multi_index.tolist()]

        pop_cv_isis = pd.DataFrame(data=0.0, index=area, columns=ind)

        for (a, l, p), r in self.pop_cv_isi.items():
            pop_cv_isis.loc[a, l+p] = r
        ax = sns.boxplot(
            data=pop_cv_isis,
            orient="h",
            palette="Set2",
        )
        col = ['blue', 'red']
        for i in range(len(ind)):
            mybox = ax.patches[i]
            mybox.set_facecolor(col[i % 2])
        plt.xlabel('Cv Isi (spikes/s)')
        plt.ylabel('Population')
        fig.savefig(os.path.join(
            self.plot_folder,
            'cv_isi.{0}'.format(extension)
        ))
        plt.clf()
        plt.close(fig)

    @_timeit
    def plot_functional_connectivity(self, save_fig=False):
        """
        Plot the functional connectivity of the network based on synaptic input currents 
        and compare it with experimental BOLD data if available.

        Parameters
        ----------
        save_fig : bool, optional
            If True, the figure will be saved to the plot folder. Default is False.
        """
        from .helpers.resting_state_networks import left_ordering

        # Define directories and load data
        data_dir = os.path.join(self.base_path, 'experimental_data/senden/rsData_7T_DKparcel/')

        # Check if experimental data exists
        exp_data_exists = (
            os.path.exists(os.path.join(data_dir, 'ROIs.txt')) and 
            os.path.exists(os.path.join(data_dir, 'rsDATA_7T_DKparcel.npy'))
        )

        if exp_data_exists:
            # Read in regions of interest.
            roi = pd.read_csv(
                os.path.join(data_dir, 'ROIs.txt'),
                header=None, names=['roi'], dtype=str
            ).squeeze()
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

            # Load clustering information
            clustering = pd.Series(left_ordering)
            ordering = clustering.keys()
            tmp_lines = clustering.values
            lines_border = np.where(tmp_lines[:-1] != tmp_lines[1:])[0]
            lines = lines_border + 1
            extended_lines = np.append(lines_border, np.array([len(left_ordering) - 1]))
            tmp = np.append(np.array([0]), extended_lines)
            points = (tmp[1:] + tmp[:-1]) * .5 + 1
            texts = clustering.iloc[extended_lines].values

            # Extract BOLD series into a dictionary of DataFrames
            exp_fc = {'lh': np.zeros((34, 34)), 'rh': np.zeros((34, 34))}

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
                
                exp_fc['lh'] += lh_fc
                exp_fc['rh'] += rh_fc

            exp_fc['lh'] /= no_of_persons
            exp_fc['rh'] /= no_of_persons
        else:
            print("No experimental data found. Only simulated data will be plotted.")

        # Figure settings
        plt.style.use('./misc/mplstyles/report_plots_master.mplstyle')
        nrows, ncols = (1, 2) if exp_data_exists else (1, 1)
        width, panel_wh_ratio = 5.63, 1.5
        height = width / panel_wh_ratio * float(nrows) / ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(width, height))
        if nrows * ncols > 1:
            axes = axes.flatten()
        else:
            axes = [axes]

        if exp_data_exists:
            # Plot experimental data
            im = axes[0].pcolormesh(exp_fc['rh'].loc[ordering][ordering], vmin=-1, vmax=1, cmap='RdYlBu_r')
            cbar = plt.colorbar(im, ax=axes[0], ticks=[-1, 0, 1])
            cbar.set_label('Correlation', rotation=270, labelpad=15)

            axes[0].set_title('Experimental FC')
            axes[0].set_xticks(points)
            axes[0].set_xticklabels(texts, rotation='vertical')
            axes[0].set_yticks(points)
            axes[0].set_yticklabels(texts)
            axes[0].axis('square')
            axes[0].invert_yaxis()
            axes[0].hlines(lines, *axes[0].get_xlim(), color='k')
            axes[0].vlines(lines, *axes[0].get_xlim(), color='k')

        # Load and plot simulated functional connectivity
        curr_in = self.synapticInputCurrent()
        synaptic_currents = curr_in.T

        # Calculate simulated functional connectivity based on synaptic
        # currents
        initial_transient = 500.
        tmax = self.sim_dict['t_sim']
        df_sim_fc_syn = synaptic_currents[
            (synaptic_currents.index >= initial_transient) & (synaptic_currents.index <= tmax)
        ].corr()
        simulated_fc = df_sim_fc_syn

        im = axes[-1].pcolormesh(simulated_fc.loc[ordering][ordering], vmin=-1, vmax=1, cmap='RdYlBu_r')

        axes[-1].set_title('Simulated FC')
        axes[-1].set_xticks(points)
        axes[-1].set_xticklabels(texts, rotation='vertical')
        axes[-1].set_yticks(points)
        axes[-1].set_yticklabels(texts)
        axes[-1].axis('square')
        axes[-1].invert_yaxis()
        axes[-1].hlines(lines, *axes[-1].get_xlim(), color='k')
        axes[-1].vlines(lines, *axes[-1].get_xlim(), color='k')

        # Save the plot
        if save_fig:
            extension = self.ana_dict['extension']
            fig.savefig(os.path.join(
                self.plot_folder,
                'simulated_synaptic_currents_correlation.{0}'.format(extension)
            ))
            plt.close(fig)
        
    @_timeit
    def calculateBOLDConnectivity(self):
        """
        Calculates BOLD connectivities
        """
        try:
            BOLD_correlation = pd.read_pickle(
                    os.path.join(self.sim_folder, 'bold_correlation.pkl')
                    )
        except FileNotFoundError:
            if not hasattr(self, 'self.BOLD'):
                self.BOLD = self.computeBOLD()
            # Start and end time, cast to [s]
            t_min = self.ana_dict['plotBOLD']['tmin'] / 1000.
            t_max = self.sim_dict['t_sim'] / 1000.

            # The BOLD output was calculated by rk45, which does not give the
            # results on a specified grid. Thus we need to sample all signals
            # to the same grid.
            bold_timesteps = self.ana_dict['plotBOLD']['stepSize'] / 1000.
            t_new = np.arange(
                    t_min,
                    t_max,
                    bold_timesteps
                    )

            BOLD_RESAMPLED = self.BOLD.copy()
            for area, data in self.BOLD.groupby(level=0):
                t = data.loc[area, 't']
                b = data.loc[area, 'bold']
                # Last value might be out of range, extrapolate it
                bold_fun = _interp1d(
                        t,
                        b,
                        bounds_error=False,
                        fill_value='extrapolate'
                        )
                b_new = bold_fun(t_new)
                BOLD_RESAMPLED.loc[area, 't'] = t_new
                BOLD_RESAMPLED.loc[area, 'bold'] = b_new

            # Calculate the correlations
            index = BOLD_RESAMPLED.loc[:, 'bold'].index
            BOLD_correlation = pd.DataFrame(
                    index=index,
                    columns=index,
                    dtype=np.float64
                    )
            for source in index:
                for target in index:
                    BOLD_correlation.loc[source, target] = np.corrcoef(
                            [
                                BOLD_RESAMPLED.loc[source, 'bold'],
                                BOLD_RESAMPLED.loc[target, 'bold']
                                ]
                            )[0,1]
            BOLD_correlation.to_pickle(
                    os.path.join(self.sim_folder, 'bold_correlation.pkl')
                    )
        return BOLD_correlation

    @_timeit
    def calculateFuncionalConnectivityCorrelations(self):
        """
        Calculates functional connectivity correlations.
        """
        if not hasattr(self, 'self.BOLD_correlation'):
            self.BOLD_correlation = self.calculateBOLDConnectivity()
        if not hasattr(self, 'self.rate'):
            self.curr_in = self.synapticInputCurrent()
        # Correlations will be written into this file
        outfile = open(os.path.join(self.ana_folder, 'correlations.txt'), 'w')
        tmin = self.ana_dict['plotConnectivities']['tmin']
        exclude_diagonal = self.ana_dict['functconn_corr']['exclude_diagonal']

        # Read in synaptic currents
        synaptic_currents = self.curr_in.T

        # Correlate synaptic currents, yields a pandas dataframe of simulated
        # functional connectivity based on synaptic input currents
        # (df_sim_fc_syn)
        df_sim_fc_syn = synaptic_currents[synaptic_currents.index >= tmin].corr()
        if exclude_diagonal:
            np.fill_diagonal(df_sim_fc_syn.values, np.nan)

        # Read in simulated functional connectivity based on calculated BOLD
        # signal
        df_sim_fc_bold = self.BOLD_correlation
        if exclude_diagonal:
            np.fill_diagonal(df_sim_fc_bold.values, np.nan)

        # Sort
        df_sim_fc_syn = df_sim_fc_syn.sort_index(axis=0).sort_index(axis=1)
        df_sim_fc_bold = df_sim_fc_bold.sort_index(axis=0).sort_index(
                axis=1
                )

        # Derive numpy array containing the correlation values. Handier than
        # the underlying pandas dataframe
        sim_fc_bold = df_sim_fc_bold.values[
            np.tril_indices(df_sim_fc_bold.shape[0])
        ]
        sim_fc_bold = sim_fc_bold[~np.isnan(sim_fc_bold)]
        sim_fc_syn = df_sim_fc_syn.values[
            np.tril_indices(df_sim_fc_syn.shape[0])
        ]
        sim_fc_syn = sim_fc_syn[~np.isnan(sim_fc_syn)]

        # Correlate simulated functional connectivity based on synaptic
        # currents and calculated BOLD signal
        sim_fc_bold__sim_fc_syn__corr = np.corrcoef(
                [sim_fc_bold, sim_fc_syn]
                )[0,1]

        # This is PEP 8 conform, but is it really nice?
        outfile.write(
                ('correlation_simulated_bold_simulated_functional_connectivity'
                f': {sim_fc_bold__sim_fc_syn__corr}\n')
                )

        try:
            # =================================================================
            # ============= READ IN AND EXTRACT EXPERIMENTAL DATA =============
            # =================================================================
            # Set Path to experimental data
            data_dir = os.path.join(
                    self.base_path, 'experimental_data', 'senden', 'rsData_7T_DKparcel'
                    )

            # Read in regions of interest.
            roi = pd.read_csv(
                os.path.join(data_dir, 'ROIs.txt'),
                header=None, names=['roi'], dtype=str
            ).squeeze()
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
                    np.fill_diagonal(lh_fc.values, np.nan)
                    np.fill_diagonal(rh_fc.values, np.nan)

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
            exp_fc__sim_fc_bold__array_lh = []
            exp_fc__sim_fc_bold__array_rh = []
            for i in range(no_of_persons):
                tmp_lh = exp_fc[i]['lh'].values[
                    np.tril_indices(exp_fc[i]['lh'].shape[0])
                ]
                tmp_rh = exp_fc[i]['rh'].values[
                    np.tril_indices(exp_fc[i]['rh'].shape[0])
                ]
                tmp_lh = tmp_lh[~np.isnan(tmp_lh)]
                tmp_rh = tmp_rh[~np.isnan(tmp_rh)]
                exp_fc_array_lh.append(tmp_lh)
                exp_fc_array_rh.append(tmp_rh)

                exp_fc__sim_fc_syn__tmp_lh = np.corrcoef(sim_fc_syn, tmp_lh)[0, 1]
                exp_fc__sim_fc_syn__tmp_rh = np.corrcoef(sim_fc_syn, tmp_rh)[0, 1]
                exp_fc__sim_fc_syn__array_lh.append(exp_fc__sim_fc_syn__tmp_lh)
                exp_fc__sim_fc_syn__array_rh.append(exp_fc__sim_fc_syn__tmp_rh)

                exp_fc__sim_fc_bold__tmp_lh = np.corrcoef(sim_fc_bold, tmp_lh)[0, 1]
                exp_fc__sim_fc_bold__tmp_rh = np.corrcoef(sim_fc_bold, tmp_rh)[0, 1]
                exp_fc__sim_fc_bold__array_lh.append(exp_fc__sim_fc_bold__tmp_lh)
                exp_fc__sim_fc_bold__array_rh.append(exp_fc__sim_fc_bold__tmp_rh)

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

            outfile.write(
                    ('exp__exp__corr_lh:\n'
                    f'{exp__exp__corr_lh}\n')
                    )
            outfile.write(
                    ('exp__exp__corr_rh:\n'
                    f'{exp__exp__corr_rh}\n')
                    )
            outfile.write(
                    ('exp_fc__sim_fc_syn__array_lh:\n'
                    f'{exp_fc__sim_fc_syn__array_lh}\n')
                    )
            outfile.write(
                    ('exp_fc__sim_fc_syn__array_rh:\n'
                    f'{exp_fc__sim_fc_syn__array_rh}\n')
                    )
            outfile.write(
                    ('exp_fc__sim_fc_bold__array_lh:\n'
                    f'{exp_fc__sim_fc_bold__array_lh}\n')
                    )
            outfile.write(
                    ('exp_fc__sim_fc_bold__array_rh:\n'
                    f'{exp_fc__sim_fc_bold__array_rh}\n')
                    )
            outfile.write(
                    ('exp__exp_mean__corr_lh: '
                    f'{exp__exp_mean__corr_lh}\n')
                    )
            outfile.write(
                    ('exp__exp_mean__corr_rh: '
                    f'{exp__exp_mean__corr_rh}\n')
                    )

            # Calculate experimental mean functional connectivity
            exp_fc_mean_lh = np.sum(
                [exp_fc[i]['lh'].values[
                        np.tril_indices(exp_fc[i]['lh'].shape[0])
                    ] for i in range(no_of_persons)],
                axis=0
                ) / no_of_persons
            exp_fc_mean_rh = np.sum(
                [exp_fc[i]['rh'].values[
                        np.tril_indices(exp_fc[i]['rh'].shape[0])
                    ] for i in range(no_of_persons)],                  
                axis=0
                ) / no_of_persons

            # Diagonal elements gave nans (correlations with themeselves)
            exp_fc_mean_lh = exp_fc_mean_lh[~np.isnan(exp_fc_mean_lh)]
            exp_fc_mean_rh = exp_fc_mean_rh[~np.isnan(exp_fc_mean_rh)]

            # Correlation experimental functional connectivity with simulated
            # functional connectivity based on synaptic currents
            exp__sim_syn__corr_lh = np.corrcoef(
                    [exp_fc_mean_lh, sim_fc_syn]
                    )[0,1]
            exp__sim_syn__corr_rh = np.corrcoef(
                    [exp_fc_mean_rh, sim_fc_syn]
                    )[0,1]

            # Correlation experimental functional connectivity with simulated
            # functional connectivity based on calculated BOLD signal
            exp__sim_bold__corr_lh = np.corrcoef(
                    [exp_fc_mean_lh, sim_fc_bold]
                    )[0,1]
            exp__sim_bold__corr_rh = np.corrcoef(
                    [exp_fc_mean_rh, sim_fc_bold]
                    )[0,1]

            outfile.write(
                    ('exp__sim_syn__corr_lh: '
                    f'{exp__sim_syn__corr_lh}\n')
                    )
            outfile.write(
                    ('exp__sim_syn__corr_rh: '
                    f'{exp__sim_syn__corr_rh}\n')
                    )
            outfile.write(
                    ('exp__sim_bold__corr_lh: '
                    f'{exp__sim_bold__corr_lh}\n')
                    )
            outfile.write(
                    ('exp__sim_bold__corr_rh: '
                    f'{exp__sim_bold__corr_rh}')
                    )
        except FileNotFoundError:
            print(('Cannot load experimantal FMRI data. '
            'Either the the datapath is wrong or the data is not available.'))
        outfile.close()

    @_timeit
    def calculateRateDistributionSimilarity(self):
        """
        Calculates the rate distribution similarity between simulated and
        experimental data.

        """
        from scipy.io import loadmat
        from scipy.stats import ks_2samp

        if not hasattr(self, 'individual_rates'):
            self.individual_rates = self.individualFiringRate()
        rates_sim = self.individual_rates

        # KS distances will be written into this file
        outfile = os.path.join(self.ana_folder, 'rate_distr_similarity.txt')

        try:
            # =================================================================
            # ============= READ IN AND EXTRACT EXPERIMENTAL DATA =============
            # =================================================================
            # Set Path to experimental data
            data_dir = os.path.join(
                    self.base_path, 'experimental_data', 'rutishauser', 'spikes'
                    )
            # Load experimental spiking data
            filename = os.path.join(data_dir, 'mfc.mat')
            data = loadmat(filename, squeeze_me=True, mat_dtype=False,
                           chars_as_strings=True)
            data = data['data_mfc']
            # Calculate rates in spks/s
            rates_data = []
            for id in range(len(data)):
                # data structure described in README.m
                stim_on = data[id]['ts']['stim_on'][()]
                baseline = data[id]['ts']['baseline_stim_on'][()]
                for st in stim_on:
                    st = np.atleast_1d(st)
                    st = st[st < baseline]
                    rates_data.append(st.size / baseline)

            # =================================================================
            # ======================= CALCULATIONS ============================
            # =================================================================
            # Extract all rates of area
            rates_sim_area = rates_sim.apply(list).groupby('area').agg(sum)
            # Remove zero rate neurons
            if self.ana_dict['rate_distribution_similarity']['remove_zeros']:
                rates_data = [rate for rate in rates_data
                              if not np.isclose(rate, 0)]
                rates_sim_area = rates_sim_area.apply(
                    lambda rates: [rate for rate in rates
                                   if not np.isclose(rate, 0)])
            # Compute KS distance
            KS_stats = rates_sim_area.apply(ks_2samp, args=(rates_data,))
            # Split tuples and write to file
            KS_stats = pd.DataFrame(data=KS_stats.values.tolist(),
                                    index=KS_stats.index,
                                    columns=['KS_dist', 'p_value'])
            KS_stats.to_csv(outfile, sep='\t')
        except FileNotFoundError:
            print(('Cannot load experimantal spiking data. '
            'Either the the datapath is wrong or the data is not available.'))

    def plotBOLDConnectivity(self):
        """
        Plots the BOLD connectivity matrix using a heatmap.If the BOLD 
        connectivity matrix has not been calculated yet, it will first 
        calculate it using the `calculateBOLDConnectivity` method.
        """
        if not hasattr(self, 'self.BOLD_correlation'):
            self.BOLD_correlation = self.calculateBOLDConnectivity()
        # Plot
        extension = self.ana_dict['extension']
        fig = plt.figure(1)
        sns.heatmap(
                self.BOLD_correlation,
                square=True,
                cmap="YlGnBu",
                cbar=True,
                cbar_kws={"shrink": .66},
                vmin=-1.,
                vmax=1.
                )
        plt.tight_layout()
        fig.savefig(os.path.join(
            self.plot_folder,
            'bold_correlation.{0}'.format(extension)
        ))
        plt.clf()
        plt.close(fig)

    def plotRasterArea(self, area):
        """
        Generates a rasterplot of the spiking activity in a specified area.
        Parameters low and high are in ms.
        ----------
        area : string
        """
        if not hasattr(self, 'popGids'):
            self.popGids = self._readPopGids()
        extension = self.ana_dict['extension']
        fraction = self.ana_dict['plotRasterArea']['fraction']
        low = self.ana_dict['plotRasterArea']['low']
        high = self.ana_dict['plotRasterArea']['high']
        fig = plt.figure(1)
        ax1 = fig.add_subplot(1, 1, 1)
        ind = []
        names = []
        gid_norm = 0
        ms_to_s = 0.001
        colors = {'E': '#1f77b4', 'I': '#ff7f0e'}
        for (layer, pop), sts in self.spikes.loc[area].items():
            # y label axis namin
            name = ' '.join([layer, pop])
            names.append(name)

            # Random shuffle spiketrains in place
            random.shuffle(sts)
            j = 0
            # Real population size, not all neurons spiked. Thus take the
            # fraction from this value.
            pop_size = self.popGids.loc[area, layer, pop].maxGID - \
                    self.popGids.loc[area, layer, pop].minGID + 1

            # Fraction of total number of neurons
            no_sts = int(fraction * pop_size)
            # Fraction of neurons that actually spiked
            frac_spiking = len(sts) * 1. / pop_size

            # where to put y label
            ind.append(- int(no_sts / 2) + gid_norm)
            # Loop as many times as we have spike trains
            for _ in range(no_sts):
                gid_norm = gid_norm - 1
                # Decide whether spiketrain contains spikes
                if random.random() < frac_spiking and j < len(sts):
                    st = sts[j]
                    j += 1
                    filtered_st = st[st > low]
                    filtered_st = filtered_st[filtered_st < high]
                    # TODO beautify plot
                    if len(filtered_st) > 0:
                        ax1.plot(
                            filtered_st * ms_to_s,
                            gid_norm * np.ones_like(filtered_st),
                            colors[pop],
                            marker='.',
                            markersize=2,
                            linestyle="None"
                        )

        ax1.axis([low * ms_to_s, high * ms_to_s, gid_norm, 0])
        ax1.set_xlabel('Time (s)')
        ax1.set_yticks(ind)
        ax1.set_yticklabels(names)
        ax1.set_title(area)
        fig.savefig(os.path.join(
            self.plot_folder,
            'raster_{0}.{1}'.format(area, extension)
        ))
        plt.clf()
        plt.close(fig)

    @_timeit
    def plot_raster_statistics(self, save_fig=False, raster_areas=None):
        """
        Plots raster statistics including raster plots for specified areas and 
        boxplots for firing rates, CV ISI, and correlation coefficients.

        Parameters
        ----------
        save_fig : bool, optional
            If True, the figure will be saved to the plot folder. Default is False.
        raster_areas : list of str, optional
            List of areas to plot raster statistics for. Default is 
            ['caudalanteriorcingulate', 'pericalcarine', 'fusiform'].
        """
        import matplotlib.gridspec as gridspec

        if raster_areas is None:
            raster_areas = ['caudalanteriorcingulate', 'pericalcarine', 'fusiform']
        
        colors = {'E': '#4c72b0ff', 'I': '#c44e52ff'}
        raster_fraction = self.ana_dict['plotRasterArea']['fraction']
        raster_low = self.ana_dict['plotRasterArea']['low']
        raster_high = self.ana_dict['plotRasterArea']['high']
        roman_to_arabic_numerals = {
            'II/III': '2/3',
            'IV': '4',
            'V': '5',
            'VI': '6',
        }
        random.seed(1234)

        # Check if necessary attributes are already loaded or need to be calculated
        if not hasattr(self, 'popGids'):
            self.popGids = self._readPopGids()
        if not hasattr(self, 'spikes'):
            self.spikes = self._readSpikes()
        if not hasattr(self, 'rate'):
            self.rate = self.meanFiringRate()
        if not hasattr(self, 'pop_lv'):
            self.pop_lv = self.popLv()
        if not hasattr(self, 'pop_cv_isi'):
            self.pop_cv_isi = self.popCvIsi()
        if not hasattr(self, 'pop_cc'):
            self.pop_cc = self.popCorrCoeff()

        # Plotting
        plt.style.use('./misc/mplstyles/report_plots_master.mplstyle')
        fig = plt.figure(constrained_layout=True, figsize=(5.5, 3.5))
        label_prms = dict(fontsize=8, fontweight='bold', va='top', ha='right')
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

        # Raster plots
        ms_to_s = 1e-3
        axs_raster = [ax_raster1, ax_raster2, ax_raster3]
        raster_labels = ['A', 'B', 'C']
        for ax, area, label in zip(axs_raster, raster_areas, raster_labels):
            ind = []
            names = []
            gid_norm = 0
            for (layer, pop), sts in self.spikes.loc[area].items():
                layer_roman = roman_to_arabic_numerals[layer]
                # Random shuffle spiketrains in place
                random.shuffle(sts)

                # Real population size, not all neurons spiked. Thus take the fraction from this value.
                popGid_alp = self.popGids.loc[area, layer, pop]
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
                    if random.random() < frac_spiking and j < len(sts):
                        st = sts[j]
                        j += 1
                        filtered_st = st[(st > raster_low) & (st < raster_high)]
                        if len(filtered_st) > 0:
                            ax.plot(
                                filtered_st * ms_to_s,
                                gid_norm * np.ones_like(filtered_st),
                                colors[pop],
                                marker='.',
                                markersize=1.5,
                                linestyle="None"
                            )

            ax.axis([raster_low * ms_to_s, raster_high * ms_to_s, gid_norm, 0])
            ax.set_xlabel('Time (s)')
            ax.set_yticks(ind)
            ax.set_yticklabels(names)
            ax.set_title(area)
            ax.text(s=label, transform=ax.transAxes, x=-0.2, y=1.06, **label_prms)

        # Boxplots
        axs_boxplots = [ax_rates, ax_cv, ax_cc]
        data_boxplots = [self.rate, self.pop_cv_isi, self.pop_cc]
        boxplots_labels = ['D', 'E', 'F']
        for ax, data, label in zip(axs_boxplots, data_boxplots, boxplots_labels):
            # reorder Series into DataFrame
            area = np.unique(data.index.get_level_values(0))
            layer = np.unique(data.index.get_level_values(1))
            pop_type = np.unique(data.index.get_level_values(2))
            multi_index = pd.MultiIndex.from_product([layer, pop_type])
            ind = [' '.join(i) for i in multi_index.tolist()]
            names = [' '.join((roman_to_arabic_numerals[l_], p_)) for l_, p_ in (i.split(' ') for i in ind)]
            data_lp = pd.DataFrame(data=np.nan, index=area, columns=ind)
            for (a, l, p), r in data.items():
                data_lp.loc[a, l+' '+p] = r

            boxplot = sns.boxplot(data=data_lp, orient='h', ax=ax, saturation=1,
                                  width=0.5, fliersize=2.5, color='k')
            col = [colors['E'], colors['I']]
            for i in range(len(ind)):
                mybox = boxplot.patches[i]
                mybox.set_facecolor(col[i % 2])
            ax.text(s=label, transform=ax.transAxes, x=-0.1, y=1.25, **label_prms)
            # Print the extension of the whiskers
            lower = []
            upper = []
            for name, x in data_lp.items():
                dat = x.dropna().values
                if len(dat) > 0:
                    upper_quartile = np.percentile(dat, 75)
                    lower_quartile = np.percentile(dat, 25)
                    iqr = upper_quartile - lower_quartile
                    upper_whisker = dat[dat <= upper_quartile + 1.5 * iqr].max()
                    lower_whisker = dat[dat >= lower_quartile - 1.5 * iqr].min()
                    lower.append(lower_whisker)
                    upper.append(upper_whisker)
            print('label:', label, 'lowest whisker:', round(min(lower), 1))
            print('label:', label, 'highest whisker:', round(max(upper), 1))
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names)
        ax_rates.set_xlim(0)
        ax_rates.set_xlabel('Firing rate (spikes/s)')
        ax_cv.set_xlim(0)
        ax_cv.set_xlabel('CV interspike interval')
        ax_cc.set_xlabel('Correlation coefficient')

        # Save figure if save_fig is True
        if save_fig:
            extension = self.ana_dict['extension']
            fig.savefig(os.path.join(self.plot_folder, f'figure_spike_statistics.{extension}'))
            plt.close(fig)

    def plot_all_binned_spike_rates_area(self):
        """
        Generates a summary plot giving an overview over the distribution of
        frequencies signal in the network.
        """
        if not hasattr(self, 'spikes_per_neuron_area_resolved'):
            self.spikes_per_neuron_population_resolved, self.spikes_per_neuron_area_resolved = self.binned_spikerates_per_neuron()
        signal = self.spikes_per_neuron_area_resolved

        # Determine max bold for y axis
        max_count = signal[:, 'hist'].apply(max).max()
        if self.ana_dict['plot_binned_spikerates_per_neuron']['max_rate']:
            max_rate = self.ana_dict['plot_binned_spikerates_per_neuron']['max_rate']
        else:
            max_rate = signal[:, 'bin_edges'].apply(max).max()

        # Set some number for number of columns
        num_col = self.ana_dict['binned_spikerates_per_neuron']['num_col']
        num_row = math.ceil(len(signal[:, 'bin_edges']) / num_col)

        # Initialize the figure
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(
                num_row,
                num_col,
                figsize=(22, 22)
                )

        # multiple line plot
        for num, (area, (tmp_count, tmp_rates)) in enumerate(signal.groupby(level=0)):
            mask = tmp_rates < max_rate
            rates = tmp_rates[mask]
            if max(tmp_rates) < max_rate:
                rates = rates[:-1]
            count = tmp_count[mask[:-1]]  # [:-1]
            # Find the right spot on the plot
            ax = axes[num // num_col][num % num_col]

            # Plot the lineplot
            ax.bar(
                    rates,
                    count,
                    width=rates[1] - rates[0],
                    edgecolor='black',
                    align='edge'
                    )

            # Same limits for everybody!
            ax.set_xlim(0, max_rate)
            ax.set_ylim(0, max_count)

            # Add title
            ax.set_title(
                    area,
                    loc='left',
                    fontsize=20,
                    fontweight=0,
                    color='black'
                    )

        fig.tight_layout()
        plt.savefig(os.path.join(
            self.plot_folder,
            'binned_spikerates_per_neuron_area_resolved.png'
            ))
        plt.clf()
        plt.close(fig)
        plt.style.use('default')

    def plotAllSynapticCurrentsSummary(self):
        """
        Generates a summary plot giving an overview over the synaptic currents
        in the network.
        """
        if not hasattr(self, 'curr_in'):
            self.curr_in = self.synapticInputCurrent()
        synaptic_currents = self.curr_in
        synaptic_currents = synaptic_currents.loc[:, synaptic_currents.columns>=500]

        # Determine max bold for y axis
        max_current = synaptic_currents.apply(max).max()

        # Set some number for number of columns
        num_col = self.ana_dict['plotAllSynapticCurrentsSummary']['num_col']
        num_row = math.ceil(len(synaptic_currents) / num_col)

        for same_axis in [True, False]:
            # Initialize the figure
            plt.style.use('seaborn-v0_8-darkgrid')
            fig, axes = plt.subplots(
                    num_row,
                    num_col,
                    figsize=(22, 22)
                    )

            # multiple line plot
            for num, (area, data_) in enumerate(synaptic_currents.groupby(level=0)):
                # Find the right spot on the plot
                ax = axes[num // num_col][num % num_col]

                times = data_.columns.values
                data = data_.values[0]

                # Plot the lineplot
                ax.plot(
                        times,
                        data,
                        marker='',
                        color='black',
                        linewidth=1.9,
                        alpha=0.9,
                        )

                # Same limits for everybody!
                ax.set_xlim(times[0], times[-1])
                if same_axis:
                    ax.set_ylim(0, max_current)

                # Not ticks everywhere
                if num < (num_row - 1)*num_col:
                    ax.tick_params(labelbottom=False)
                else:
                    ax.set_xlabel('Time (s)')

                if num % num_col != 0:
                    ax.tick_params(labelleft=False)
                else:
                    ax.set_ylabel('Synaptic current')

                # Add title
                ax.set_title(
                        area,
                        loc='left',
                        fontsize=20,
                        fontweight=0,
                        color='black'
                        )

            fig.tight_layout()
            plt.savefig(os.path.join(
                self.plot_folder,
                f'synaptic_currents_overview_same_axis_{same_axis}.png'
                ))
            plt.clf()
            plt.close(fig)
            plt.style.use('default')

    def plotAllBOLDSignalSummary(self):
        """
        Generates a summary plot giving an overview over the BOLD signal
        in the network.
        """
        if not hasattr(self, 'BOLD'):
            self.BOLD = self.computeBOLD()

        # Determine max bold for y axis
        max_bold = self.BOLD[:, 'bold'].apply(max).max()

        # Set some number for number of columns
        num_col = self.ana_dict['plotAllBOLDSignalSummary']['num_col']
        num_row = math.ceil(len(self.BOLD[:, 'bold']) / num_col)

        # Initialize the figure
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(
                num_row,
                num_col,
                figsize=(22, 22)
                )

        # multiple line plot
        for num, (area, (data, times)) in enumerate(self.BOLD.groupby(level=0)):
            # Find the right spot on the plot
            ax = axes[num // num_col][num % num_col]

            # Plot the lineplot
            ax.plot(
                    times,
                    data,
                    marker='',
                    color='black',
                    linewidth=1.9,
                    alpha=0.9,
                    )

            # Same limits for everybody!
            ax.set_xlim(times[0], times[-1])
            ax.set_ylim(0, max_bold)

            # Not ticks everywhere
            if num < (num_row - 1)*num_col:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel('Time (s)')

            if num % num_col != 0:
                ax.tick_params(labelleft=False)
            else:
                ax.set_ylabel('BOLD')

            # Add title
            ax.set_title(
                    area,
                    loc='left',
                    fontsize=20,
                    fontweight=0,
                    color='black'
                    )

        fig.tight_layout()
        plt.savefig(os.path.join(
            self.plot_folder,
            'bold_overview.png'
            ))
        plt.clf()
        plt.close(fig)
        plt.style.use('default')

    def plotAllFiringRatesSummary(self):
        """
        Generates a summary plot giving an overview over the spiking activity
        in the network. Each subplot is the time course of the spike rates
        averaged across all neurons in a given area.
        """
        if not hasattr(self, 'popGids'):
            self.popGids = self._readPopGids()
        if not hasattr(self, 'rate_hist_areas'):
            self.rate_hist, self.rate_hist_areas = self.firingRateHistogram()
        rate_hist_areas = self.rate_hist_areas * 1000.  # ms to s
        resolution = self.ana_dict['rate_histogram']['binsize']

        if self.ana_dict['rate_histogram']['t_start']:
            t_start = self.ana_dict['rate_histogram']['t_start']
        else:
            t_start = 0.

        if self.ana_dict['rate_histogram']['t_stop']:
            t_stop = self.ana_dict['rate_histogram']['t_stop']
        else:
            t_stop = self.sim_dict['t_sim']

        rate_hist_areas = rate_hist_areas.apply(lambda x: x[int(t_start/resolution):])

        times = np.arange(t_start, t_stop, resolution) / 1000.  # To seconds

        # Determine max rate for y axis
        max_rate = rate_hist_areas.apply(max).max()

        # Set some number for number of columns
        num_col = self.ana_dict['plotAllFiringRatesSummary']['num_col']
        num_row = math.ceil(len(rate_hist_areas) / num_col)

        # Initialize the figure
        plt.style.use('seaborn-v0_8-darkgrid')

        for same_y_axis in [False, True]:
            fig, axes = plt.subplots(
                    num_row,
                    num_col,
                    figsize=(22, 22)
                    )

            # multiple line plot
            for num, (area, data) in enumerate(rate_hist_areas.items()):
                # Find the right spot on the plot
                ax = axes[num // num_col][num % num_col]

                # Plot the lineplot
                ax.plot(
                        times,
                        data,
                        marker='',
                        color='black',
                        linewidth=1.9,
                        alpha=0.9,
                        )

                # Same limits for everybody!
                ax.set_xlim(times[0], times[-1])
                if same_y_axis:
                    ax.set_ylim(0, max_rate)
                else:
                    ax.set_ylim(0, max(data))

                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Rate (Spikes / s)')

                # Add title
                ax.set_title(
                        area,
                        loc='left',
                        fontsize=20,
                        fontweight=0,
                        color='black'
                        )

            fig.tight_layout()
            plt.savefig(os.path.join(
                self.plot_folder,
                'spiking_overview_same_y_axis_{}.png'.format(same_y_axis)
                ))
            plt.clf()
            plt.close(fig)
        plt.style.use('default')

    @_timeit
    def _readSpikes(self):
        """
        Reads SpikeTrains of the simulation.

        Returns
        -------
        spikes : Series of lists of SpikeTrains
        """
        try:
            spikes = pd.read_pickle(
                os.path.join(self.sim_folder, 'spikes.pkl')
            )
        except FileNotFoundError:
            print('Loading SpikeTrains from dat')
            spikes = self._readSpikesFromDAT()
            # Save spikes to pickle for faster read access
            spikes.to_pickle(os.path.join(self.sim_folder, 'spikes.pkl'))
        return spikes

    @_timeit
    def _readPopGids(self):
        """
        Reads the min / max GID for each population.

        Returns
        -------
        popGids : DataFrame
            Columns ['minGID', 'maxGID']
        """
        from ast import literal_eval
    

        try:
            popGids = pd.read_pickle(
                os.path.join(self.sim_folder, 'population_GIDs.pkl')
            )
        except FileNotFoundError:
            popGids = pd.read_csv(
                os.path.join(self.sim_folder, 'population_GIDs.dat'),
                sep=';', header=None, names=['minGID', 'maxGID']
            )
            popGids.index = pd.MultiIndex.from_tuples([
                literal_eval(s) for s in popGids.index
            ])
            popGids['pop_size'] = popGids['maxGID'] - popGids['minGID'] + 1
            popGids.index.names = ['area', 'layer', 'pop']
            # Save popGids to pickle for faster read access
            popGids.to_pickle(
                os.path.join(self.sim_folder, 'population_GIDs.pkl')
            )
        return popGids

    def _readSpikesFromDAT(self):
        """
        Reads spikes from dat output files using pandas.
        Stores all SpikeTrains for one population in a list wich in
        turn is contained in a Series.

        Returns
        -------
        spikes : Series of arrays of arrays containing spike timings.
        """
        import glob
        import time
        import subprocess
        import multiprocessing
        from multiprocessing import Pool

        python_sort = self.ana_dict['python_sort']
        if python_sort:
            # Read in population gids
            popGids = self.popGids
            # glob all spikes files
            dat_files = glob.glob(os.path.join(self.sim_folder, 'spikes', '*.dat'))
            # Read in all dat files into a big dataframe with two columns, gid and
            # t. Magically this seems to work in parallel
            spikes = pd.concat(
                    (
                        pd.read_csv(
                            f,
                            sep='\t',
                            skiprows=3,
                            index_col=False,
                            header=None,
                            names=['gid', 't']
                            ) for f in dat_files
                    ),
                    ignore_index=True
                    )
            # Sort spikes, first gid, then t
            spikes = spikes.sort_values(['gid', 't'])
            # Create a spiketrain (i.e. numpy arrays) for every gid and store them
            # in a cell
            spikes = spikes.groupby('gid').apply(
                    lambda group: group.t.values
                    ).reset_index()
            spikes.columns = ['gid', 't']

            no_spiking = []
            for row in popGids.itertuples():
                area = row.Index[0]
                layer = row.Index[1]
                pop = row.Index[2]
                first_gid = row.minGID
                last_gid = row.maxGID
                try:
                    first = np.where(
                            (spikes.gid <= last_gid) & (spikes.gid >= first_gid)
                            )[-1][0]
                    last = np.where(
                            (spikes.gid <= last_gid) & (spikes.gid >= first_gid)
                            )[-1][-1]
                    spikes.loc[first:last, 'area'] = area
                    spikes.loc[first:last, 'layer'] = layer
                    spikes.loc[first:last, 'pop'] = pop
                except IndexError:
                    no_spiking.append([area, layer, pop])
                    pass
            spikes = spikes.groupby(
                    ['area', 'layer', 'pop']
                    ).apply(lambda group: group.t.values)

            for (area, layer, pop) in no_spiking:
                spikes.loc[area, layer, pop] = np.array([])
            spikes = spikes.sort_index()
            return spikes
        else:
            # This routine should be used when the simulated time is long, eg
            # 100 seconds. It is not faster when analyzing 10 seconds of
            # biological time. But when we simulate long and thus generate a
            # lot of data we need to do a lot of sorting. And pandas is bad at
            # this. In fact the original routine won't even finish sorting the
            # data for 100 seconds of biological time in 24 h. So for analyzing
            # larger datasets we need a more efficient spike reading in
            # routine. This routine relies a lot on GNU coreutils. E.g. sort
            # can be parallelized and it implements mergesort.
            # Note: this routine implicitly assumes that every area has spiked

            # ==============================================================================
            #                                   Definitions
            # ==============================================================================

            sorted_fn = 'combined_and_sorted_spiketrains.txt'
            available_cores = multiprocessing.cpu_count()

            popGids = self.popGids.sort_values('minGID')

            # ==============================================================================
            #                                 Presorting data
            #
            # Here I presort all dat files. Presorted files can easily be merged without
            # memory constraints. Also mergesort is quite fast.
            # The reason I use GNU sort instead of sorting in python is that GNU sort also
            # works in parallel. GNU sort uses 8 cores per default, which seems be a good
            # value as a rule of thumb. Furthermore I sort as many files as possible
            # simultaneously. I can launch available_cores / 8 jobs at a given time.
            # But: I haven't compared python sort vs GNU sort
            #
            # This step takes, for 100 seconds of bio time on 64 cores, 15 minutes
            # ==============================================================================

            file_ending = '*.dat'
            self.rec_folder = os.path.join(self.sim_folder, 'spikes')
            dat_files = glob.glob(os.path.join(self.rec_folder, file_ending))

            ts = time.time()
            pool = Pool(math.ceil(available_cores / 8))  # Gnu sort sorts in parallel with 8 threads
            _ = pool.map(shell_presort_all_dat, dat_files)
            te = time.time()
            passed_time = round(te - ts, 3)
            print(f'presorting data took {passed_time} s')

            # ==============================================================================
            #                                 mergesorting data
            #
            # In this step the data from before is mergesorted into a huge file. This file
            # contains all spikes, sorted by gid and time. GNU sort is used as it provides
            # a good mergesorting algorithm. This is way I used it instead of a pythonic
            # way. But: I haven't compared python sort vs GNU sort
            #
            # This step takes, for 100 seconds of bio time on 64 cores, 22 minutes
            # The resulting file, for 100 seconds of time, is 120 GB
            # ==============================================================================

            ts = time.time()
            subprocess.check_output(
                    f'export LC_ALL=C; sort -k1,1n -k2,2n -m --parallel=8 {self.rec_folder}/*_sorted.txt > {self.rec_folder}/{sorted_fn}',
                    shell=True
                    )
            te = time.time()
            passed_time = round(te - ts, 3)
            print(f'mergesorting data took {passed_time} s')

            # ==============================================================================
            #            Splitting sorted data into population resolved data files
            #
            # Now we have this gigantic sorted datafile (120GB) which we would like to
            # split into population specific files. Meaning: Every text file contains all
            # spikes of the given population in a sorted manner. Splitting such a big file
            # takes quite some time. The algorithm works in the following way:
            # 1) Find a population in the middle of the popGids Dataframe
            # 2) Split according to this population
            # 3) Now we have two dataframes, one is left and the other one is right. These
            #    Dataframes also need to be splitted. As they are independent, they can be
            #    split in parallel
            # 4) As the algorithm progresses, the chunks become more numerous (more
            #    parallelization) and smaller (faster).
            # 5) When all has been split, we are done :)
            #
            # I haven't benchmarked the GNU splitting routine against a pyhtonic approach
            #
            # This step takes, for 100 seconds of bio time on 64 cores, 80 minutes
            # ==============================================================================

            all_tmp = [(popGids, sorted_fn, 0, self.rec_folder)]

            for iteration in range(math.ceil(math.log2(len(popGids)))):
                ts = time.time()
                cores = min(int(2**iteration), available_cores)
                pool = Pool(cores)
                tmp = pool.starmap(split_files, all_tmp)
                all_tmp = []
                if tmp:
                    for x in range(len(tmp)):
                        if tmp[x] and len(tmp[x]) > 0:
                            all_tmp.append((tmp[x][0], tmp[x][1], tmp[x][4]+1, self.rec_folder))
                            if len(tmp[x]) > 3:
                                all_tmp.append((tmp[x][2], tmp[x][3], tmp[x][4]+1, self.rec_folder))
                te = time.time()
                passed_time = round(te - ts, 3)
                print(f'Splitting iteration {iteration} took {passed_time} s')

            # ==============================================================================
            # Rename files such that the filename contains exact information on population
            #
            # We rename all files such that the filename gives away which population we are
            # looking at.
            #
            # This step is fast, around 1 minute
            # ==============================================================================

            d_population_forward = {
                    'II/III': '23',
                    'IV': '4',
                    'V': '5',
                    'VI': '6'
                    }

            d_population_backward = {
                    '23': 'II/III',
                    '4': 'IV',
                    '5': 'V',
                    '6': 'VI'
                    }

            ts = time.time()
            file_ending = sorted_fn + '_*'
            dem_files = glob.glob(os.path.join(self.rec_folder, file_ending))
            for fn in dem_files:
                with open(fn) as f:
                    a = int(f.readline().split()[0])
                    exact_population_tmp = popGids[(popGids.minGID <= a) & (popGids.maxGID >= a)].index.tolist()[0]
                    exact_population = (exact_population_tmp[0], d_population_forward[exact_population_tmp[1]], exact_population_tmp[2])

                    # Assert that first and last spike are from the same population and
                    # that every sorting so far has done the correct thing.
                    check = int(subprocess.check_output(['tail', '-1', fn]).split()[0])
                    check_population = popGids[(popGids.minGID <= check) & (popGids.maxGID >= check)].index.tolist()[0]
                    assert exact_population_tmp == check_population

                    new_name = os.path.join(
                            self.rec_folder,
                            '_'.join(exact_population) + '.spiketrains'
                            )
                    os.rename(fn, new_name)
            te = time.time()
            passed_time = round(te - ts, 3)
            print(f'Assigning correct names to datafiles took {passed_time} s')

            # ==============================================================================
            #          Pull spikefiles together, have one spiketrain per line, remove gids
            #
            # Before the text files look like this:
            #
            # 7 5.5
            # 7 8.6
            # 7 9.2
            # 8 5.7
            # 8 6.5
            #
            # We now make sure that every line contains a spiketrain and remove the gid
            # number as it is not important anymore. The result looks like this:
            #
            # 5.5 8.6 9.2
            # 5.7 6.5
            #
            # This step takes, for 100 seconds of bio time on 8 cores, 20 minutes. on 64
            # cores probably 3 minutes.
            # ==============================================================================

            ts = time.time()
            file_ending = '*.spiketrains'
            dem_files = glob.glob(os.path.join(self.rec_folder, file_ending))
            pool = Pool(available_cores)
            _ = pool.map(shell_spiketrainify, dem_files)
            te = time.time()
            passed_time = round(te - ts, 3)
            print(f'spiketrainify data took {passed_time} s')

            # ==============================================================================
            #                        Read files into pandas Dataframes
            #
            # Reading in 100 seconds of bio time takes 20 minutes. Saving takes 1.5 minutes.
            # Note: This probably can be optimized
            # ==============================================================================

            ts = time.time()
            file_ending = '*.cut'
            spikes_txt = glob.glob(os.path.join(self.rec_folder, file_ending))
            all_spikes = pd.Series(index=popGids.index, dtype=object)
            for spike_txt_file in spikes_txt:
                area, layer, pop, *_ = spike_txt_file.split('/')[-1].replace('.','_').split('_')
                layer = d_population_backward[layer]
                tmp = []
                with open(spike_txt_file) as f:
                    lines=f.readlines()
                    for line in lines:
                        tmp.append(np.fromstring(line, dtype=float, sep=' '))
                all_spikes[(area, layer, pop)] = np.array(tmp, dtype=object)
            te = time.time()
            passed_time = round(te - ts, 3)
            print(f'putting data into Series took {passed_time} s')

            ts = time.time()
            all_spikes.to_pickle(os.path.join(self.ana_folder, 'spikes.pkl'))
            te = time.time()
            passed_time = round(te - ts, 3)
            print(f'saving Series took {passed_time} s')
            return all_spikes

    def getHash(self):
        """
        Creates a hash from analysis parameters.

        Returns
        -------
        hash : str
            Hash for the simulation
        """
        from dicthash import dicthash

        hash_ = dicthash.generate_hash_from_dict(self.ana_dict)
        return hash_

    def dump(self, base_folder):
        """
        Exports the full analysis specification. Creates a subdirectory of
        base_folder from the analysis hash where it puts all files.

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
        fn = os.path.join(out_folder, 'ana.yaml')
        with open(fn, 'w') as outfile:
            yaml.dump(self.ana_dict, outfile, default_flow_style=False)


def cvIsi(sts, t_start=None, t_stop=None, CV_min_spikes=2, take_mean=True):
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
    import numpy as np

    # ensure that there are only spiketrains of len > 1. This is
    # for np.diff which needs to output at least one value in order
    # for np.std and np.mean to return something which is not
    # np.nan. Otherwise return 0.
    if t_start:
        sts = [st[st >= t_start] for st in sts]
    if t_stop:
        sts = [st[st <= t_stop] for st in sts]
    sts = [st for st in sts if len(st) >= CV_min_spikes]
    if len(sts) > 0:
        isi = [np.diff(x, 1) for x in sts]
        cv = [np.std(x) / np.mean(x) for x in isi]
        if take_mean:
            return np.mean(cv)
        else:
            return cv
    return 0.


def calculate_lv(isi, t_ref):
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
    import numpy as np

    # NOTE Elephant and mam use different functions. Elephant uses the normal
    # local variation whereas mam uses the revised local variation. LV depends
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


def LV(sts, t_ref, t_start=None, t_stop=None, LV_min_spikes=3, take_mean=True):
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
    import numpy as np

    # ensure that there are only spiketrains of len > 2.
    # So every spiketrain st in sts has len(st) > 1.
    if t_start:
        sts = [st[st >= t_start] for st in sts]
    if t_stop:
        sts = [st[st <= t_stop] for st in sts]
    sts = [st for st in sts if len(st) >= LV_min_spikes]
    if len(sts) > 0:
        isi = [np.diff(x, 1) for x in sts]
        lvr = [calculate_lv(x, t_ref) for x in isi]
        if take_mean:
            return np.mean(lvr)
        else:
            return lvr
    return 0.


def calc_rates(sts, sim_dict, ana_dict):
    """
    Calculates the histogram of rates of a list of spiketrains sts.
    NOTE: The units of the returned rates are in spikes / ms

    Returns
    -------
    rate : np.ndarray
        array of binned rates
    """
    import numpy as np

    resolution = ana_dict['rate_histogram']['binsize']
    t_min = 0.
    t_max = ana_dict['rate_histogram']['t_stop']
    if t_min is None:
        t_min = 0.
    if t_max is None:
        t_max = sim_dict['t_sim']
    num_bins = int((t_max - t_min) / resolution)
    if len(sts) > 0:
        # Gives same output as:
        # elstat.time_histogram(
        #     sts, binsize=resolution*pq.ms, output='counts'
        # )
        counts, _ = np.histogram(
                np.concatenate(sts).ravel(),
                bins=num_bins,
                range=(t_min, t_max)  # or range=(t_min + resolution / 2., t_max + resolution / 2.)
                )
        rate = counts * 1. / resolution
        return rate
    return np.zeros(num_bins)


def correlation(sts, ana_dict, sim_dict):
    """
    Calculates the correlation coefficients for a subset of neurons for a
    list of spiketrains.
    Taken from correlation toolbox, available from
    https://github.com/INM-6/correlation-toolbox .

    Parameters
    ----------
    ana_dict : dictionary
        dictionary containing values

    Returns
    -------
    cc : float
        Correlation coefficient
    """
    import numpy as np

    subsample = ana_dict['correlation_coefficient']['subsample']
    _, hist = instantaneous_spike_count(sts, ana_dict, sim_dict)
    rates = strip_binned_spiketrains(hist)[:subsample]
    # Need at least 2 spiketrains
    if len(rates) > 1:
        cc = np.corrcoef(rates)
        cc = np.extract(1-np.eye(cc[0].size), cc)
        cc[np.where(np.isnan(cc))] = 0.
        return np.mean(cc)
    return 0.


def instantaneous_spike_count(data, ana_dict, sim_dict):
    '''
    Create a histogram of spike trains.
    Taken from correlation toolbox, available from
    https://github.com/INM-6/correlation-toolbox .

    Parameters
    ----------
    ana_dict : dictionary
        dictionary containing values

    Returns
    -------
    bins : np.array
        Bins
    hist : np.array
        Histogram
    '''
    import numpy as np

    tbin = ana_dict['correlation_coefficient']['tbin']
    tmin = ana_dict['correlation_coefficient']['tmin']
    tmax = ana_dict['correlation_coefficient']['tmax']
    if tmin is None:
        tmin = 0.
    if tmax is None:
        tmax = sim_dict['t_sim']
    assert(tmin < tmax)
    bins = np.arange(tmin, tmax + tbin, tbin)
    hist = np.array([np.histogram(x, bins)[0] for x in data])
    return bins[:-1], hist


def strip_binned_spiketrains(sp):
    '''
    Removes binned spiketrains which do not contain a single spike
    Taken from correlation toolbox, available from
    https://github.com/INM-6/correlation-toolbox .

    Parameters
    ----------
    sp : np.array
        Array containing histogram.

    Returns
    -------
    sp_stripped : np.array
        Binned spiketrains with empty spiketrains removed.
    '''
    import numpy as np

    sp_stripped = np.array(
            [x for x in sp if abs(np.max(x) - np.min(x)) > 1e-16]
            )
    return sp_stripped

def shell_presort_all_dat(fn):
    """
    Sort all .dat files in the folder.

    Parameters
    ----------
    fn : str
        The filename to be processed.
    """
    import subprocess

    # -n +4 is important for dat files as they contain a header
    # subprocess.check_output(
    #         f'export LC_ALL=C; f={fn}; tail -n +4 ${{f}} | sort -k1,1n -k2,2n --parallel=8 > ${{f%.dat}}_sorted.txt',
    #         shell=True
    #         )
    subprocess.check_output(
            f'export LC_ALL=C; f={fn}; tail -n +1 ${{f}} | sort -k1,1n -k2,2n --parallel=8 > ${{f%.dat}}_sorted.txt',
            shell=True
            )

def shell_spiketrainify(fn):
    """
    This function is used to sort and organize the spiking data into
    spiketrains per neuron.

    Parameters
    ----------
    fn : str
        The filename to be processed.
    """
    import subprocess

    lol2 = '''
    awk '
    {
      if($1==k)
      printf("%s"," ")
      else {
          if(NR!=1)
          print ""
        printf("%s\t",$1)
      }
      for(i=2;i<NF;i++)
        printf("%s ",$i)
      printf("%s",$NF)
      k=$1
    }
    END{
    print ""
    }' '''
    lol3 = f'{fn} > {fn}.sorted'
    lol4 = f'''; cut -d$'\t' -f2 {fn}.sorted > {fn}.sorted.cut'''
    # lol5 = f'; rm {fn} {fn}.sorted'
    lol = lol2 + lol3 + lol4  # + lol5

    subprocess.check_output(
            lol,
            shell=True
            )

def split_files(df, fn, iteration, rec_folder):
    """
    The function uses awk and csplit commands to split the file based 
    on the maximum group ID of the left half of the dataframe. It also 
    deletes the original file if it is not the initial one.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to be split.
    fn : str
        The name of the file that contains the dataframe.
    iteration : int
        The number of times the function has been called recursively.
    rec_folder : str
        The path of the folder where the files are stored.

    Returns
    -------
    df_left : pandas.DataFrame
        The left half of the dataframe.
    fn_left : str
        The name of the file that contains the left half of the dataframe.
    df_right : pandas.DataFrame
        The right half of the dataframe.
    fn_right : str
        The name of the file that contains the right half of the dataframe.
    iteration : int
        The updated iteration number.
    """
    import subprocess

    if len(df) > 1:
        where_to_split = math.floor(len(df)/2)
        df_left = df.iloc[:where_to_split]
        df_right = df.iloc[where_to_split:]
        maxGID = df_left.iloc[-1].maxGID

        iteration_ = str(iteration)
        fn_left = fn + f'_left_{iteration_}'
        fn_right = fn + f'_right_{iteration_}'
        subprocess.check_output(
                f'''a=$(awk '$1>{maxGID}{{print NR, $0; exit}}' {rec_folder}/{fn} | cut -d ' ' -f1); csplit -sf {rec_folder}/part.{fn}. {rec_folder}/{fn} $a; mv {rec_folder}/part.{fn}.00 {rec_folder}/{fn_left}; mv {rec_folder}/part.{fn}.01 {rec_folder}/{fn_right}''',
                shell=True
                )
        if fn != 'all_sorted_spiketrains2.txt':
            subprocess.check_output(f'rm {rec_folder}/{fn}', shell=True)
        return df_left, fn_left, df_right, fn_right, iteration

def kernel_for_psc(tau_s, dt):
    """
    Calculates the kernel used in the calculation of postsynaptic currents.

    Parameters
    ----------
    tau_s : float
        Synaptic time constant
    dt : float
        Simulation resolution

    Returns
    -------
    kernel : np.array
    """
    import numpy as np

    # Calculate exponential kernel for PSCs
    t_ker = np.arange(-10*tau_s, 10*tau_s, dt)
    kernel = np.exp(- t_ker / tau_s) / tau_s
    kernel[t_ker < -dt/2] = 0  # Make filter causal
    kernel /= kernel.sum()  # Normalize filter
    return kernel

def analysisDictFromDump(dump_folder):
    """
    Creates a analysis dict from the files created by Analysis.dump().

    Parameters
    ----------
    dump_folder : string`
        Folder with dumped files

    Returns
    -------
    ana_dict : dict
        Full analysis dictionary
    """
    import os
    import yaml

    fn = os.path.join(dump_folder, 'ana.yaml')
    with open(fn, 'r') as ana_file:
        ana_dict = yaml.load(ana_file)
    return ana_dict
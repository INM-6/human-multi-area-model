"""
Provides default parameters used in the analysis.
"""

__all__ = [
    "params",
]

params = {}

params['mplstyles'] = 'misc/mplstyles/report_plots_master.mplstyle'
params['extension'] = 'png'
params['seed'] = 2903
params['python_sort'] = True

params['rate_histogram'] = {
    'binsize': 1.,    # in ms
    't_start': 500.,  # in ms
    't_stop': None    # in ms
}

params['meanFiringRate'] = {
    't_start': 500.,  # in ms
    't_stop': None    # in ms
}

params['individualFiringRate'] = {
    't_start': 500.,  # in ms
    't_stop': None    # in ms
}

params['correlation_coefficient'] = {
    'subsample': 2000,
    'tbin': 1.,    # in ms
    'tmin': 500.,  # in ms
    'tmax': None   # in ms
}

params['compute_louvain'] = {
    'tmin': 500.  # in ms
}

params['cv'] = {
    't_start': 500.,  # in ms
    't_stop': None,   # in ms
    'min_spikes': 10
}

params['lvr'] = {
    't_start': 500.,  # in ms
    't_stop': None,   # in ms
    'min_spikes': 10
}

params['functconn_corr'] = {
    'exclude_diagonal': True
}

params['plotAllFiringRatesSummary'] = {
    'num_col': 4
    }

params['plotAllSynapticCurrentsSummary'] = {
    'num_col': 4
    }

params['plotAllBOLDSignalSummary'] = {
    'num_col': 4
    }

params['plotConnectivities'] = {
    'tmin': 500.  # in ms
}

params['plotBOLD'] = {
    'tmin': 500.,   # in ms
    'stepSize': 1.  # in ms
}

params['plotRasterArea'] = {
    'fraction': 0.03,
    'low': 500,
    'high': 750
}

params['binned_spikerates_per_neuron'] = {
    'bins': 250,
    'num_col': 4
}

params['rate_distribution_similarity'] = {
    'remove_zeros': True,
}

params['plot_binned_spikerates_per_neuron'] = {
    'max_rate': 40.
}

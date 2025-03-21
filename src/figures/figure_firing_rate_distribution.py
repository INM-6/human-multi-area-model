import os
from os.path import join as path_join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_firing_rate_distribution(filepath, figname):
    """
    Plot firing rate distribution for each population across different areas.

    Parameters
    ----------
    filepath : str
        Path to the directory containing the pickle files.
    figname : str
        Name of the output figure.
    """

    # Read the rates_individual pickle file
    rates_individual = pd.read_pickle(path_join(filepath, 'rates_individual.pkl'))

    # Plot 8 figures in a 4x2 grid the rates of each population across areas
    fig, ax = plt.subplots(4, 2, figsize=(5.63, 6), sharex=True, sharey=True)  # Increase figure size for readability

    row, col = 0, 0  # Initialize subplot counters
    layer_labels = ['2/3', '4', '5', '6']
    binsize = 0.5

    for layer in rates_individual.index.get_level_values(1).unique():
        for pop in rates_individual.index.get_level_values(2).unique():
            # Initialize list to store firing rate distributions
            rates_dist = []

            # Calculate firing rate distribution for each area
            for area in rates_individual.index.get_level_values(0).unique():
                if (layer, pop) not in rates_individual[area]:
                    print('Skipping ' + area + ' ' + layer + ' ' + pop)
                    continue
                else:
                    # Use logarithmic bins, but compute the histogram on log10 of the rates
                    vals, bins = np.histogram(np.log10(rates_individual[area, layer, pop]), bins=np.arange(-3, 5, binsize))
                    if np.sum(vals) > 0:
                        vals = vals / (binsize * np.sum(vals))
                    rates_dist.append(vals)

            # Calculate mean and standard deviation of firing rate distributions
            mean_rates_count = np.mean(rates_dist, axis=0)
            std_rates_count = np.std(rates_dist, axis=0)

            # Convert log10 bins to actual firing rates (10^bins)
            actual_bins = 10**bins[:-1]

            ax[row, col].plot(actual_bins, mean_rates_count, linewidth=2, color='black')
            ax[row, col].fill_between(actual_bins, mean_rates_count - std_rates_count, mean_rates_count + std_rates_count, alpha=0.5, color='black')
            ax[row, col].set_title(layer_labels[row] + ' ' + pop)

            # Set x-axis to logarithmic scale to represent firing rates correctly
            ax[row, col].set_xscale('log')
            ax[row, col].set_xlim(1e-3, 1e4)  # Adjust limits as necessary for your data
            # ax[row, col].set_ylim(-0.5, 1.6)

            col += 1
            if col >= 2:
                col = 0
                row += 1

    # Add overall figure title
    fig.supylabel('Density')
    fig.supxlabel('Firing rate (spikes/s)')

    fig.tight_layout()
    fig.savefig(f'figures/{figname}.pdf')


# Set the filepath and figname
filepath = os.path.join(os.getcwd(), 'out/8c49a09f51f44fbb036531ce0719b5ba/4772f0b020c9f3310f4096a6db758343/')
figname = "figure_firing_rate_distribution_bestfit"

# Call the plot_firing_rate_distribution function
plot_firing_rate_distribution(filepath, figname)

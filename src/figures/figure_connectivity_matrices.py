"""
This script generates and plots connectivity matrices for both downscaled and full-scale networks.
It calculates the indegrees by dividing the number of synapses by the number of neurons and groups 
the subpopulations together by summing the indegrees for each area.
The resulting matrices are visualized using matplotlib and can be saved to a specified file path.
This script is primarily used in the humam_tutorial.ipynb notebook.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from ..data_loader.dk_fullnames_to_shortnames import dk_full_to_short

def calculate_indegrees(synapses, neurons):
    """
    Calculate the indegrees by dividing synapses by neuron numbers.
    Group the subpopulations together by summing the indegrees for each area.
    """
    indegrees = synapses.div(neurons, axis=0)
    grouped_indegrees = indegrees.groupby(level=0).sum().T.groupby(level=0).sum().T
    return grouped_indegrees

def plot_matrix(ax, matrix, title, areas_short_names):
    """
    Plot the connectivity matrix using pcolor.
    """
    c = ax.pcolor(matrix, cmap='YlGn', norm=LogNorm())
    plt.colorbar(c, ax=ax, label='Indegree')
    ax.set_title(title)
    ax.set_xlabel('Source Areas')
    ax.set_ylabel('Target Areas')
    ax.set_xticks(np.arange(0.5, len(areas_short_names), 1))
    ax.set_xticklabels(areas_short_names, rotation=90)
    ax.set_yticks(np.arange(0.5, len(areas_short_names), 1))
    ax.set_yticklabels(areas_short_names)

def plot_connectivity_matrices(net_params_downscaled, net_params_fullscale=None, save_path=None):
    """
    Plots the connectivity matrices for both downscaled and full-scale networks.
    Parameters:
    net_params_downscaled (dict): Dictionary containing the downscaled network parameters, including synapses and neuron numbers.
    net_params_fullscale (dict or None): Dictionary containing the full-scale network parameters, including synapses and neuron numbers.
    save_path (str or None): File path to save the figure. If None, the figure will not be saved.
    Returns:
    None: This function does not return any value. It displays the connectivity matrices using matplotlib.
    """
    if net_params_fullscale is None:
        # Only plot the downscaled connectivity matrix
        fig, ax = plt.subplots(figsize=(6, 5))
        
        # Calculate the indegrees for the downscaled version
        grouped_indegrees_downscaled = calculate_indegrees(
            net_params_downscaled['synapses_internal'], 
            net_params_downscaled['neuron_numbers']
        )
        
        # Rename the areas to their short names
        areas_short_names = [dk_full_to_short[area] for area in grouped_indegrees_downscaled.columns]
        
        # Plot the downscaled connectivity matrix
        plot_matrix(ax, grouped_indegrees_downscaled, 'Downscaled Connectivity Matrix', areas_short_names)
        
    else:
        # Create a figure with 1 row and 2 columns
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        
        # Calculate the indegrees for the downscaled version
        grouped_indegrees_downscaled = calculate_indegrees(
            net_params_downscaled['synapses_internal'], 
            net_params_downscaled['neuron_numbers']
        )
        
        # Rename the areas to their short names
        areas_short_names = [dk_full_to_short[area] for area in grouped_indegrees_downscaled.columns]
        
        # Plot the downscaled connectivity matrix
        plot_matrix(axs[0], grouped_indegrees_downscaled, 'Downscaled Connectivity Matrix', areas_short_names)
        
        # Get the number of synapses and neurons for the full-scale network
        full_scale_synapses = net_params_fullscale['SN']
        full_scale_neurons = net_params_fullscale['NN']
        
        # Calculate the indegrees for the full-scale version
        grouped_indegrees_full_scale = calculate_indegrees(full_scale_synapses, full_scale_neurons)
        
        # Plot the full-scale connectivity matrix
        plot_matrix(axs[1], grouped_indegrees_full_scale, 'Full-scale Connectivity Matrix', areas_short_names)
        
    plt.tight_layout()
    
    # Save the figure if save_path is provided
    if save_path:
        plt.savefig(save_path)
    
    plt.show()
import os
from os.path import join as path_join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..network import networkDictFromDump

figname = "figure_mult_and_autapses_count"
net_dict = networkDictFromDump(os.path.join(os.getcwd(), 'out/8c49a09f51f44fbb036531ce0719b5ba/'))

# ========== Preprocess ==========
area_list = net_dict['area_list']
NN = net_dict['neuron_numbers']
synapses = net_dict['synapses_internal']

# Estimate the number of autapses per neuron averaged per area
autapses = pd.DataFrame(index=area_list, columns=['mean autapses'])
for area in area_list:
    df = synapses.loc[area, area]
    df = df.div(NN[area]**2, axis=1)
    autapses.loc[area, 'mean autapses'] = df.values.mean()

# Number of multapses given an existing synapse
area_list = synapses.index.unique(level=0).tolist()
layer_list = ['II/III', 'IV', 'V', 'VI']
pop_list = ['E', 'I']
layer_pop_list = ['II/III_E', 'II/III_I', 'IV_E', 'IV_I', 'V_E', 'V_I', 'VI_E', 'VI_I']
multapses = pd.DataFrame(data=0, index=area_list, columns=[layer_pop_list])
for target_area in area_list:
    for target_layer in layer_list:
        for target_pop in pop_list:
            count = 0.
            Nt = NN[(target_area, target_layer, target_pop)] # number of target neurons
            for source_area in area_list:
                for source_layer in layer_list:
                    for source_pop in pop_list:
                        Ns = NN[(source_area, source_layer, source_pop)] # number of source neurons
                        nsyn = synapses.loc[(target_area, target_layer, target_pop), (source_area, source_layer, source_pop)] # number of synapses
                        if nsyn > 0:
                            # remove autapses when source and target are the same
                            if target_area == source_area and target_layer == source_layer and target_pop == source_pop:
                                p_unique_conn = 1.- (1. - 1./(Nt*(Nt-1.)))**(nsyn-nsyn/Nt)
                                multapses.loc[target_area, target_layer+'_'+target_pop] += (nsyn-nsyn/Nt)/(p_unique_conn*Nt*(Nt-1.))
                            else:
                                p_unique_conn = 1.- (1. - 1./(Ns*Nt))**nsyn
                                multapses.loc[target_area, target_layer+'_'+target_pop] += nsyn/(p_unique_conn*Ns*Nt)
                            count += 1.
            if count > 0:
                multapses.loc[target_area, target_layer+'_'+target_pop] /= count

# =========== Plot ==========
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6., 7.), sharex=True)

autapses.plot(kind='bar', y='mean autapses', color='MediumSeagreen', edgecolor='black', ax=ax1, legend=False)
ax1.set_ylabel('Autapses per neuron')

multapses[multapses>0].mean(axis=1).plot(kind='bar', color='MediumSeagreen', edgecolor='black', ax=ax2, legend=False)
ax2.set_ylabel('Average multapse degree')

# Add text labels
ax1.text(-0.15, 1.0, '(A)', ha='left', va='top', transform=ax1.transAxes, fontsize=12, fontweight='bold')
ax2.text(-0.15, 1.0, '(B)', ha='left', va='top', transform=ax2.transAxes, fontsize=12, fontweight='bold')

fig.tight_layout()
fig.savefig(f'figures/{figname}.pdf')
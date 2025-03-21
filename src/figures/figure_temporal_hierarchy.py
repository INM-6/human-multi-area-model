""" 
Generates the matrix part of the Temporal Hierarchy figure, plus shows 
the hierarchy in 3d by opening a link to the Scalable Brain Atlas in a
browser window.

Author: Rembrandt Bakker & other authors from the publications
https://doi.org/10.1101/2023.03.23.533968
and
https://doi.org/10.1371/journal.pcbi.1006359
""" 

"""
python module imports
"""
import os
from os.path import join as path_join

import copy
import json
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import median_abs_deviation

# to organize areas by hierarchy
from scipy.optimize import minimize

# to show brain in Scalable Brain Atlas Composer
import webbrowser

"""
global parameters
"""
filepath = os.path.join(os.getcwd(), 'out/8c49a09f51f44fbb036531ce0719b5ba/4772f0b020c9f3310f4096a6db758343/7f5b8ecbfce8346314b050b58bbab1cf/')
filename = 'rate_histogram_areas.pkl'
spikeRateDataPickleFile = path_join(filepath, filename)
diagnosticPlots = False

# What part of the simulation to use (12500 ms available)
t_total = 12500
t_start = 2500 # to ignore the transient
t_stop = t_total

# gross ordering of the brain into four lobes
right_ordering_four_lobes = {
    # Occipital Lobe
    'cuneus': 'occipital',
    'lateraloccipital': 'occipital',
    'lingual': 'occipital',
    'pericalcarine': 'occipital',

    # Temporal Lobe
    'superiortemporal': 'temporal',
    'middletemporal': 'temporal',
    'inferiortemporal': 'temporal',
    'entorhinal': 'temporal',
    'parahippocampal': 'temporal',
    'temporalpole': 'temporal',
    'transversetemporal': 'temporal',
    'bankssts': 'temporal',
    'fusiform': 'temporal',  # Note: fusiform can be considered part of both occipital and temporal lobes
    'insula': 'temporal',  # Included with temporal lobe in this coarse division


    # Parietal Lobe
    'precuneus': 'parietal',
    'supramarginal': 'parietal',
    'inferiorparietal': 'parietal',
    'superiorparietal': 'parietal',
    'postcentral': 'parietal',
    'posteriorcingulate': 'parietal',  # Often associated with parietal lobe functions
    'isthmuscingulate': 'parietal',  # Can be considered part of parietal lobe

    # Frontal Lobe
    'medialorbitofrontal': 'frontal',
    'lateralorbitofrontal': 'frontal',
    'parsopercularis': 'frontal',
    'parsorbitalis': 'frontal',
    'parstriangularis': 'frontal',
    'rostralmiddlefrontal': 'frontal',
    'caudalmiddlefrontal': 'frontal',
    'superiorfrontal': 'frontal',
    'frontalpole': 'frontal',
    'precentral': 'frontal',
    'paracentral': 'frontal', # check this one
    'rostralanteriorcingulate': 'frontal',  # Associated with frontal lobe functions
    'caudalanteriorcingulate': 'frontal'  # Associated with frontal lobe functions
}

right_ordering = right_ordering_four_lobes

"""
Load area-averaged firing rates, based on spike trains from a model run
"""
rate_per_area = pd.read_pickle(spikeRateDataPickleFile)
rate_per_area = rate_per_area*1000 # convert to Hz

# Remove the transient
for area in rate_per_area.index:
    rate_per_area[area] = rate_per_area[area][t_start:t_stop]

time = range(len(rate_per_area.iloc[0]))

# Area list
area_list = rate_per_area.index

# Area-averaged firing rates as a matrix
rate_values = np.array(rate_per_area.tolist())

"""
Routine to select a peak from the cross correlation function.
Positive/negative peaks are treated separately.
Smaller (absolute) delays are preferred, larger delays are only accepted
if peak is more than a standard deviation taller.
"""
# source for confidence intervals: https://faculty.ksu.edu.sa/sites/default/files/probability_and_statistics_for_engineering_and_the_sciences.pdf
def selectPeak(cc,lags,sd):
    peaks, props = signal.find_peaks(cc,height=sd,distance=5)
    def sortFunc(i):
        return np.abs(lags[i])
    bestPeaks = {}
    bestHeights = {}
    for delaySign in ['neg','pos']:
        if delaySign == 'pos':
            sortedPeaks = sorted(peaks[lags[peaks]>=0],key=sortFunc)
        else: 
            sortedPeaks = sorted(peaks[lags[peaks]<0],key=sortFunc)
            
        bestPeak = np.nan
        bestHeight = np.nan
        if len(sortedPeaks):
            idx = sortedPeaks[0]
            bestPeak = lags[idx]
            bestHeight = cc[idx]
            for idx in sortedPeaks[1:]:
                if cc[idx]>bestHeight+sd:
                    bestPeak = lags[idx]
                    bestHeight = cc[idx]
                prev = idx
        bestPeaks[delaySign] = bestPeak
        bestHeights[delaySign] = bestHeight
    return bestPeaks,bestHeights

"""
Compute the cross-correlation functions between all pairs of brain areas
"""
max_lag = 50
peak_matrix = np.zeros((len(area_list), len(area_list)))
for i,area in enumerate(area_list):
    print(area)
    diagnosticPlots = ()
    for j,other_area in enumerate(area_list):
        signal1 = rate_per_area[area]
        signal1 = signal1-signal1.mean()
        signal1 /= signal1.std()
        signal2 = rate_per_area[other_area]
        signal2 = signal2-signal2.mean()
        signal2 /= signal2.std()
        
        N = len(signal1)
        nChunks = 9
        chunkSize = N//nChunks

        # Now divide the timeseries into 9 chunks 
        # and compute the crosscorrelation function for each chunk 
        cc_chunks = []
        for c in range(nChunks):
            # autocorrelation method
            chunk1 = signal1[c*chunkSize:(c+1)*chunkSize]
            chunk1 = (chunk1-chunk1.mean())/chunk1.std()
            chunk2 = signal2[c*chunkSize:(c+1)*chunkSize]
            chunk2 = (chunk2-chunk2.mean())/chunk2.std()
            cc = signal.correlate(chunk1, chunk2, mode='full')/chunkSize
            lags = signal.correlation_lags(chunk1.size, chunk2.size, mode='full')
            lag_indices = np.where((lags <= max_lag) & (lags >= -max_lag))
            cc = cc[lag_indices]
            cc_chunks.append(cc)

        # Also compute the crosscorrelation function for the full data
        cc_full = signal.correlate(signal1, signal2, mode='full')/N
        lags = signal.correlation_lags(signal1.size, signal2.size, mode='full')
        lag_indices = np.where((lags <= max_lag) & (lags >= -max_lag))
        cc_full = cc_full[lag_indices]
        lags_full = lags[lag_indices]

        # Use the 9 chunks to compute the standard deviation of the cross-correlation functions
        cc_chunks_np = np.array(cc_chunks)
        cc_chunks_np -= cc_full # cc_full is used as an estimator for the average of cc_chunks
        cc_chunk_std = cc_chunks_np.flatten().std()
        
        # The standard deviation of the full data cross correlation function is lower than that of the chunks
        cc_full_std = cc_chunk_std/np.sqrt(nChunks)
        
        bestPeaks_full,bestHeights_full = selectPeak(cc_full,lags_full,cc_full_std)
        if diagnosticPlots:
            fig,ax = plt.subplots(1,1,figsize=[10,6])
            ax.plot(lags_full, cc_full, label=f'{area} vs. ' + other_area, color='orange',linewidth=5)
            ax.vlines(x=bestPeaks_full['neg'], ymin=cc_full.min(), ymax=cc_full.max(), color="orange",linewidth=5)
            ax.vlines(x=bestPeaks_full['pos'], ymin=cc_full.min(), ymax=cc_full.max(), color="orange",linewidth=5)

        # Independently compute the best negative/best positive delay peak for each chunk
        bestNegPeak_chunks = []
        bestPosPeak_chunks = []
        for c in range(nChunks):
            cc = cc_chunks[c]
            lags = signal.correlation_lags(chunkSize, chunkSize, mode='full')
            lag_indices = np.where((lags <= max_lag) & (lags >= -max_lag))
            lags = lags[lag_indices]
            bestPeaks,bestHeights = selectPeak(cc,lags,cc_chunk_std)

            bestNegPeak_chunks.append(bestPeaks['neg'])
            bestPosPeak_chunks.append(bestPeaks['pos'])
            if diagnosticPlots:
                ax.plot(lags, cc, label=f'{area} vs. ' + other_area, color='steelblue')
                ax.vlines(x=bestPeaks['neg'], ymin=cc.min(), ymax=cc.max(), color="firebrick")
                ax.vlines(x=bestPeaks['pos'], ymin=cc.min(), ymax=cc.max(), color="green")
        
        medNeg = np.median(bestNegPeak_chunks)
        medPos = np.median(bestPosPeak_chunks)
        madNeg = median_abs_deviation(bestNegPeak_chunks)
        madPos = median_abs_deviation(bestPosPeak_chunks)

        # Decide which peaks are significant:
        # 1. the peaks obtained for the chunks should not be too divergent,
        # 2. the median delay of the peak for the chunks should be close to the estimated delay of the full data
        # 3. the delay should not exceed 30
        peak_matrix[i][j] = None
        acceptNeg = madNeg <= 4 and np.abs(bestPeaks_full['neg']-medNeg) <= 3 and np.abs(bestPeaks_full['neg'])<30
        acceptPos = madPos <= 4 and np.abs(bestPeaks_full['pos']-medPos) <= 3 and np.abs(bestPeaks_full['pos'])<30
        if acceptNeg and acceptPos:
            if bestHeights_full['neg']>bestHeights_full['pos']+2*cc_full_std:
                # negative delay peak is significantly higher than positive delay peak
                peak_matrix[i][j] = bestPeaks_full['neg']
            elif bestHeights_full['pos']>bestHeights_full['neg']+2*cc_full_std:
                # positive delay peak is significantly higher than negative delay peak
                peak_matrix[i][j] = bestPeaks_full['pos']
            else:
                peak_matrix[i][j] = None # no clear largest peak, undecided
        else:
            if acceptNeg:
                peak_matrix[i][j] = bestPeaks_full['neg']
            if acceptPos:
                peak_matrix[i][j] = bestPeaks_full['pos']

        if diagnosticPlots:
            mn = cc_full.min()
            mx=cc_full.max()
            mn = mn-0.5*(mx-mn)
            mx = mx+0.5*(mx-mn)
            if not acceptNeg:
                ax.plot([lags[0],0,np.nan,lags[0],0],[mn,mx,np.nan,mx,mn], color='red',linewidth=5)
            if not acceptPos:
                ax.plot([0,lags[-1],np.nan,0,lags[-1]],[mn,mx,np.nan,mx,mn], color='red',linewidth=5)

            ax.set_xlabel("Time delay (s)")
            ax.set_ylabel("Cross correlation (-)")
            ax.set_title(f'{area} vs {other_area}: delay {peak_matrix[i][j]}, {madNeg}+{madPos}')
            plt.show()
    
            plt.close(fig)

peak_matrix = pd.DataFrame(peak_matrix, index=area_list, columns=area_list)

"""
Routines for Hierarchical Clustering
"""
def dev(i, j, hierarchy, cc_matrix):
    """
    Deviation function for the linear programming algorithm
    determining the hierarchy.
    """
    if np.isnan(cc_matrix[i][j]):
        return 0
    else:
        return (hierarchy[i] - hierarchy[j] - cc_matrix[i][j])

def hier_dev(hierarchy, cc_matrix):
    deviation = 0.
    for i in range(hierarchy.size):
        for j in range(hierarchy.size):
            deviation += (dev(i, j, hierarchy, cc_matrix)) ** 2
    return np.sqrt(deviation)

def create_hierarchy(cc_matrix, areas):
    """
    Determined the hierarchy for a given set of areas and their
    cross-correlation peak matrix.
    """
    res = minimize(hier_dev, np.random.rand(
        cc_matrix[0].size), args=(cc_matrix,))
    hierarchy = res['x']
    index_transformation = np.argsort(hierarchy)
    hierarchical_areas = copy.copy(areas)
    hierarchical_areas = np.array(hierarchical_areas)
    hierarchical_areas = hierarchical_areas[index_transformation]
    hierarchy = hierarchy[index_transformation]
    # Map hierarchy onto [0,1] interval
    hierarchy -= np.min(hierarchy)
    hierarchy /= np.max(hierarchy)
    return res, hierarchy, hierarchical_areas, index_transformation

"""
Perform Hierarchical Clustering
"""
cc_matrix = peak_matrix.to_numpy() # Rows: Source area, Cols: Target area
res, hierarchy, hierarchical_areas, index_transformation = create_hierarchy(cc_matrix, area_list)

hierarchy_to_area_list = []
for area in area_list:
    hierarchy_to_area_list.append(
        np.where(hierarchical_areas == area)[0][0]
    )

cc_matrix_hier = cc_matrix[index_transformation][:, index_transformation]
cc_matrix_hier_masked = np.ma.masked_where(np.isnan(cc_matrix_hier), cc_matrix_hier)
    
"""
Plot the Hierarchical Clustering matrix, with brain areas on the x- and y-axes,
x-symbols at NaN-value locations
"""
# Define colors for each group of brainregions
group_colors = {
    'frontal': '#AAFFAA', # green
    'occipital': '#EEFFEE', # light grey
    'temporal': '#BBBBFF', # blue
    'parietal': '#DDDDDD', # darker grey
}

colormap = 'seismic'

fig, ax = plt.subplots(figsize=(8, 6.5))

ax.set_xlabel('Source area\n$\longrightarrow$ temporal hierarchy $\longrightarrow$')
ax.set_ylabel('$\longrightarrow$ temporal hierarchy $\longrightarrow$\nTarget area')

vlim = np.nanmax(np.abs(peak_matrix))
im = ax.pcolormesh(
    cc_matrix_hier_masked[:, :],
    cmap = colormap,
    vmin = -vlim,
    vmax = vlim 
    )

ax.set_xticks(np.arange(0, len(hierarchical_areas), 1) + 0.5)
ax.set_yticks(np.arange(0, len(hierarchical_areas), 1) + 0.5)

ax.set_xticklabels(hierarchical_areas[:], rotation='vertical')
ax.set_yticklabels(hierarchical_areas[:])

# Add background color to tick labels
xtick_labels = ax.get_xticklabels()
ytick_labels = ax.get_yticklabels()

for label in xtick_labels:
    area = label.get_text()
    group = right_ordering[area]
    label.set_backgroundcolor(group_colors[group])
    label.set_bbox(dict(facecolor=group_colors[group], edgecolor='none', pad=0))  # Add padding

for label in ytick_labels:
    area = label.get_text()
    group = right_ordering[area]
    label.set_backgroundcolor(group_colors[group])
    label.set_bbox(dict(facecolor=group_colors[group], edgecolor='none', pad=0))  # Add padding

xNan,yNan = np.nonzero(np.isnan(cc_matrix_hier[:, :]))
plt.plot(yNan+0.5,xNan+0.5,'x',color='gray')
plt.plot(np.arange(len(area_list))+0.5,np.arange(len(area_list))+0.5,'.',color='gray')

cbar = plt.colorbar(im, ax=ax)
cbar.set_label(r'Delay, positive when source leads target ($\mathrm{ms}$)')

plt.savefig("figures/figure_temporal_hierarchy_matrix.pdf")

# Prepare urls to display hierarchies in SBA computer
rescale = matplotlib.colors.Normalize(vmin=0, vmax=len(hierarchical_areas)-1)
cdict = {'red':   [[0.0,  1.0, 1.0],
                   [1.0,  0.0, 0.0]],
         'green': [[0.0,  1.0, 1.0],
                   [1.0, 0.0, 0.0]],
         'blue':  [[0.0,  1.0, 1.0],
                   [1.0,  0.2, 0.2]]}
mycmap = LinearSegmentedColormap('testCmap', segmentdata=cdict, N=256)

urls = {}
for surface in ['gm','infl','wm']:
   coloredRegions = { f'{source}(L,{surface})':[to_hex(mycmap(rescale(j)))[1:],1] for j,source in enumerate(hierarchical_areas)}
   coloredRegions[f'unknown(L,{surface})'] = ["000000",1]
   url = f'https://neuroinformatics.nl/sba-alpha/www/composer/?template=WKBetal10&scene={{"regions":{json.dumps(coloredRegions)},"background":"ffffff"}}'
   url = url.replace('"','%22')
   urls[surface] = url
   
# Show hierarchy in SBA Composer as whitematter (use wm), grey matter (use gm) or inflated (use infl) surface
webbrowser.open(urls['infl'])


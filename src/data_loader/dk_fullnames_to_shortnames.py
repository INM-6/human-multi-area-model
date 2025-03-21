# -*- coding: utf-8 -*-

"""
Mapping of the full Names of the
Desikan & Killiany regions to their abbreviations.

Original Paper describing the regions of interest:
https://surfer.nmr.mgh.harvard.edu/ftp/articles/desikan06-parcellation.pdf

Following the abbreviation system found in
https://github.com/DevelopmentalImagingMCRI/freesurfer_statsurf_display/blob/master/freesurfer_statsurf_plot.m
"""

__all__ = [
    "dk_full_to_short",
]

dk_full_to_short = {
    u'bankssts': u'BSTS', # banks of superior temporal sulcus
    u'caudalanteriorcingulate': u'CAC', # Caudal anterior-cingulate cortex
    u'caudalmiddlefrontal': u'CMF', # Caudal middle frontal gyrus
    u'cuneus': u'CUN', # Cuneus cortex
    u'entorhinal': u'ENT', # Entorhinal cortex, Found: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5433913/
    u'fusiform': u'FUS', # Fusiform gyrus
    u'inferiorparietal': u'INFP', # Inferior parietal cortex
    u'inferiortemporal': u'IT', # Inferior temporal gyrus
    u'isthmuscingulate': u'ISTC', # Isthmus–cingulate cortex
    u'lateraloccipital': u'LOCC', # Lateral occipital cortex
    u'lateralorbitofrontal': u'LORB', # Lateral orbital frontal cortex
    u'lingual': u'LIN', # Lingual gyrus
    u'medialorbitofrontal': u'MORB', # Medial orbital frontal cortex
    u'middletemporal': u'MT', # Middle temporal gyrus
    u'parahippocampal': u'PARH', # Parahippocampal gyrus
    u'paracentral': u'PARC', # Paracentral lobule
    u'parsopercularis': u'POPE', # Pars opercularis
    u'parsorbitalis': u'PORB', # Pars orbitalis
    u'parstriangularis': u'PTRI', # Pars triangularis
    u'pericalcarine': u'PCAL', # Pericalcarine cortex
    u'postcentral': u'PSTS', # Postcentral gyrus
    u'posteriorcingulate': u'PC', # Posterior-cingulate cortex
    u'precentral': u'PREC', # Precentral gyrus
    u'precuneus': u'PCUN', # Precuneus cortex
    u'rostralanteriorcingulate': u'RAC', # Rostral anterior cingulate cortex
    u'rostralmiddlefrontal': u'RMF', # Rostral middle frontal gyrus
    u'superiorfrontal': u'SF', # Superior frontal gyrus
    u'superiorparietal': u'SP', # Superior parietal cortex
    u'superiortemporal': u'ST', # Superior temporal gyrus
    u'supramarginal': u'SMAR', # Supramarginal gyrus
    u'frontalpole': u'FP', # Frontal pole
    u'temporalpole': u'TP', # Temporal pole
    u'transversetemporal': u'TT', # Transverse temporal cortex
    u'insula': u'INS', # Insula
}

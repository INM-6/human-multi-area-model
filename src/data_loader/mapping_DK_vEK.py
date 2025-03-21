# -*- coding: utf-8 -*-

"""
Mapping from the Desikan & Killiany regions
to the areas in von Economo & Koskinas
based on (Goulas et al. bioRxiv 2016, Table 1).

Returns
-------
dk_to_vEK : dict
    Mapping dictionary
"""

__all__ = [
    "dk_to_vEK",
]

dk_to_vEK = {
    u'bankssts': [u'PH'],
    u'caudalanteriorcingulate': [u'LA1', u'LA2'],
    u'caudalmiddlefrontal': [u'FC', u'FB'],
    u'cuneus': [u'OA', u'OB'],
    u'entorhinal': [u'HA', u'HB'],
    u'fusiform': [u'TF'],
    u'inferiorparietal': [u'PG'],
    u'inferiortemporal': [u'TE'],
    u'isthmuscingulate': [u'LD', u'LC2'],
    u'lateraloccipital': [u'OA'],
    u'lateralorbitofrontal': [u'FG'],
    u'lingual': [u'OA', u'OB', u'PH'],
    u'medialorbitofrontal': [u'FH', u'FL'],
    u'middletemporal': [u'TE'],
    u'parahippocampal': [u'HC', u'HD'],
    u'paracentral': [u'PA', u'PB1', u'PB2', u'PC'],
    u'parsopercularis': [u'FCBm'],
    u'parsorbitalis': [u'FF'],
    u'parstriangularis': [u'FDt'],  # TODO Is u'FDt' really FDGamma?
    u'pericalcarine': [u'OC'],
    u'postcentral': [u'PC'],
    u'posteriorcingulate': [u'LC1', u'LC2', u'LC3'],
    u'precentral': [u'FA'],
    u'precuneus': [u'PE'],
    u'rostralanteriorcingulate': [u'LA1', u'LA2'],
    u'rostralmiddlefrontal': [u'FD\u0394', u'FD', u'FC'],
    u'superiorfrontal': [u'FC', u'FD', u'FB'],
    u'superiorparietal': [u'PE'],
    u'superiortemporal': [u'TA'],
    u'supramarginal': [u'PF'],
    u'frontalpole': [u'FE'],
    u'temporalpole': [u'TG'],
    u'transversetemporal': [u'TC', u'TD'],
    u'insula': [u'IA', u'IB']
}

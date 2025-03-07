# Assignment of areas to resting state networks. Based on:
# Kabbara, A., EL Falou, W., Khalil, M. et al.
# The dynamic functional core network of the human brain at rest.
# Sci Rep 7, 2936 (2017). https://doi.org/10.1038/s41598-017-03420-6
#
# acronyms:
# default mode network (DMN), dorsal attentional network (DAN),
# salience network (SAN), auditory network (AUD), visual network (VIS)

__all__ = [
    "left_ordering",
    "right_ordering",
]

left_ordering = {'isthmuscingulate': 'DMN',
                 'medialorbitofrontal': 'DMN',
                 'posteriorcingulate': 'DMN',
                 'precuneus': 'DMN',
                 'rostralanteriorcingulate': 'DMN',
                 'lateralorbitofrontal': 'DMN',
                 'parahippocampal': 'DMN',
                 'caudalanteriorcingulate': 'DAN',
                 'inferiortemporal': 'DAN',
                 'middletemporal': 'DAN',
                 'parsopercularis': 'DAN',
                 'parsorbitalis': 'DAN',
                 'parstriangularis': 'DAN',
                 'insula': 'SAN',
                 'rostralmiddlefrontal': 'SAN',
                 'supramarginal': 'SAN',
                 'caudalmiddlefrontal': 'SAN',
                 'superiortemporal': 'AUD',
                 'cuneus': 'VIS',
                 'lateraloccipital': 'VIS',
                 'fusiform': 'VIS',
                 'lingual': 'VIS',
                 'bankssts': 'other',
                 'entorhinal': 'other',
                 'frontalpole': 'other',
                 'inferiorparietal': 'other',
                 'superiorfrontal': 'other',
                 'paracentral': 'other',
                 'pericalcarine': 'other',
                 'postcentral': 'other',
                 'precentral': 'other',
                 'superiorparietal': 'other',
                 'temporalpole': 'other',
                 'transversetemporal': 'other'}

right_ordering = {'isthmuscingulate': 'DMN',
                  'medialorbitofrontal': 'DMN',
                  'posteriorcingulate': 'DMN',
                  'precuneus': 'DMN',
                  'rostralanteriorcingulate': 'DMN',
                  'lateralorbitofrontal': 'DMN',
                  'parahippocampal': 'DMN',
                  'caudalanteriorcingulate': 'DMN',
                  'inferiortemporal': 'DAN',
                  'middletemporal': 'DAN',
                  'parsopercularis': 'DAN',
                  'parsorbitalis': 'DAN',
                  'parstriangularis': 'DAN',
                  'insula': 'SAN',
                  'rostralmiddlefrontal': 'SAN',
                  'supramarginal': 'SAN',
                  'caudalmiddlefrontal': 'SAN',
                  'superiortemporal': 'AUD',
                  'cuneus': 'VIS',
                  'lateraloccipital': 'VIS',
                  'fusiform': 'VIS',
                  'lingual': 'VIS',
                  'bankssts': 'other',
                  'entorhinal': 'other',
                  'frontalpole': 'other',
                  'inferiorparietal': 'other',
                  'superiorfrontal': 'other',
                  'paracentral': 'other',
                  'pericalcarine': 'other',
                  'postcentral': 'other',
                  'precentral': 'other',
                  'superiorparietal': 'other',
                  'temporalpole': 'other',
                  'transversetemporal': 'other'}

# -*- coding: utf-8 -*-

"""
Provides the binzegger & mohan data.

Returns
-------
binzegger : DataFrame
    Probability of having a cell body given the synapse location
    according to the Binzegger data.
mohan : DataFrame
    Probability of having a cell body given the synapse location
    according to the Mohan et al. data.
"""

import pandas as pd

__all__ = [
    "binzegger",
    "mohan",
]

binzegger = pd.DataFrame(
    data=[
        [0.567082554283337, 0, 0, 0, 0],
        [0, 0.16304434207349616, 0, 0, 0],
        [0.17328245714496945, 0.8369556579265038, 0.7258933731905455, 0, 0],
        [0, 0, 0.15914115512152516, 0, 0],
        [0.2568327220638478, 0, 0.026277562606296354, 0.760757867472445, 0],
        [0, 0, 0, 0.10406630543342058, 0],
        [0.002802266507845749, 0, 0.08868790908163299, 0.13517582709413395, 0.8533038600077738],
        [0, 0, 0, 0, 0.14669613999222625]
    ],
    index=pd.MultiIndex.from_product(
        [['II/III', 'IV', 'V', 'VI'], ['E', 'I']],
        names=['layer', 'population']
    ),
    columns=['I', 'II/III', 'IV', 'V', 'VI']
)

# Mohan data: only pyramidal cells -> dummy probability of 0 for inhibitory neurons
# Numbers extracted from experimental_data/mohan/dendriteLength.py (see the README therein)
# The array corresponds to `s2cb` which is plotted at the end of the script.
mohan = pd.DataFrame(
    data=[
        [0.9278774811692967, 0.909748357973469, 0.4162649934194968, 0.0, 0.0],
        [0, 0, 0, 0, 0],
        [0.0668572994545314, 0.083453428814039, 0.490646804597804, 0.1497274751515097, 0.0],
        [0, 0, 0, 0, 0],
        [0.005265219376171895, 0.005428066629491999, 0.061860804254694896, 0.4769364895274995, 0.0954296854918162],
        [0, 0, 0, 0, 0],
        [0.0, 0.0013701465830000094, 0.031227397728004324, 0.37333603532099074, 0.9045703145081838],
        [0, 0, 0, 0, 0]
    ],
    index=pd.MultiIndex.from_product(
        [['II/III', 'IV', 'V', 'VI'], ['E', 'I']],
        names=['layer', 'population']
    ),
    columns=['I', 'II/III', 'IV', 'V', 'VI']
)

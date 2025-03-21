# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

__all__ = [
    "VonEconomoKoskinas",
]

class VonEconomoKoskinas():
    """
    Loads the structural data of von Economo & Koskinas.
    Provided by Alex Goulas and Claus Hilgetag.


    Parameters
    ----------
    file_path : string
        Path to data file
    """

    def __init__(self, file_path):
        self.file_path = file_path

        # Load data into DataFrame
        self.data = pd.read_excel(
            self.file_path,
            sheet_name='Sheet1',
            index_col=[0, 2]
        )

        # Keep only cell density and layer thickness & rename
        self.data = self.data[[
            u"mean cell content per layer [cells/mm³]",
            u"Layer thickness overall [mm]"
        ]]
        self.data = self.data.rename(columns={
            u"mean cell content per layer [cells/mm³]": "neuron_density",
            u"Layer thickness overall [mm]": "layer_thickness"
        })

        # Cast and replace not castable
        self.data.where(self.data != '-', 0, inplace=True)
        self.data.where(self.data != '60.000 (?)', 60000, inplace=True)
        self.data['layer_thickness'] = self.data['layer_thickness'].astype(
            np.float64)
        self.data['neuron_density'] = self.data['neuron_density'].astype(
            np.float64)

        # Fill nas due to cell merge in layer thickness by 0
        self.data["layer_thickness"] = self.data["layer_thickness"].fillna(
            value=0
        )
        # Fill nas due to cell merge in neuron density by previous value
        self.data["neuron_density"] = self.data["neuron_density"].fillna(
            value=0
        )

    def __repr__(self):
        return ('Von Economo & Koskinas structural data\n'
                'File: {0}\nAreas (index lvl 0):\n{1}\n'
                'Layers (index lvl 1):\n{2}\nColumns:\n{3}\n'
                .format(self.file_path,
                        self.data.index.levels[0].values,
                        self.data.index.levels[1].values,
                        self.data.columns.values)
                )

    def getNeuronDensity(self):
        """
        Returns the neuron density in cells/mm^3.

        Returns
        -------
        neuron_density : Series
        """
        return self.data['neuron_density']

    def getLayerthickness(self):
        """
        Returns the layer thickness in mm.

        Returns
        -------
        layer_thickness : Series
        """
        return self.data['layer_thickness']

    def getAreaList(self):
        """
        Returns the list of areas.

        Returns
        -------
        area_list : list
        """
        return self.data.index.levels[0].values

    def dump(self, file_path, key=''):
        """
        Save all data as a json file
        """
        # TODO keep as hdf?
        self.data.to_hdf(file_path, key='von Economo Koskinas', mode='w')


if __name__ == "__main__":

    vek_path = '../experimental_data/voneconomokoskinas/StructuralData_VonEconomoKoskinas.xls'
    vek = VonEconomoKoskinas(vek_path)
    print(vek)
    print(vek.getNeuronDensity())
    print(vek.getLayerthickness())
    print(vek.getAreaList())
    # vek.dump('vek_tmp.hdf')
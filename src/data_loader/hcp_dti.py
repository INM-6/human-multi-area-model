# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.io import loadmat

__all__ = [
    "HcpDesikanKilliany",
    "VolumesDK",
]

class HcpDesikanKilliany():
    """
    Connectivity from the HCP in the Desikan Killiany atlas.
    Provided by Alex Goulas and Claus Hilgetag.

    Returns
    -------
    data : Panel
        All available data
    names : list
        Names of Desikan Killiany atlas
    """

    def __init__(self, file_path):
        self.file_path = file_path

        # Read in data
        self.data = loadmat(self.file_path)

        # Preprocess names: Convert to 1d list & strip leading 'ctx-lh-'
        self.names = self.data['DK_Names']
        self.names = np.concatenate(self.names.flatten()).tolist()
        self.names = [s.split('-')[-1] for s in self.names]

        self.dist = pd.DataFrame(
                self.data['Dist'],
                columns=self.names,
                index=self.names,
                dtype=np.float64
                )

        self.RightHem = pd.DataFrame(
            self.data['LowResDK_EstimatedDist_RHem'],
                columns=self.names,
                index=self.names,
                dtype=np.float64
                )

        self.LeftHem = pd.DataFrame(
            self.data['LowResDK_EstimatedDist_LHem'],
                columns=self.names,
                index=self.names,
                dtype=np.float64
                )

        self.ConnectivityRight = pd.DataFrame(
            self.data['C_R'],
                columns=self.names,
                index=self.names,
                dtype=np.float64
                )

        self.ConnectivityLeft = pd.DataFrame(
            self.data['C_L'],
                columns=self.names,
                index=self.names,
                dtype=np.float64
                )

    def __repr__(self):
        return ('HCP connectivity data\n'
                'File: {0}\nRegions (major & minor axis):\n{1}\n'
                'Items:\n{2}\n'
                .format(self.file_path,
                        np.array(self.names),
                        self.data.items.values)
                )

    def getDesikanKillianyNames(self):
        """
        Returns the names of the different Desikan Killiany areasself.

        Returns
        -------
        names : list
        """
        return self.names

    def getDist(self):
        """
        Returns the euclidean distances in mm of the different Desikan Killiany
        areas.

        Returns
        -------
        euclidean_distance : DataFrame
        """
        return self.dist

    def getFiberLengthRight(self):
        """
        Returns the fiber lengthes in mm of the right hemisphere of the different
        Desikan Killiany areas.

        Returns
        -------
        fiber_length_right : DataFrame
        """
        return self.RightHem

    def getFiberLengthLeft(self):
        """
        Returns the fiber lengthes in mm of the left hemisphere of the different
        Desikan Killiany areas.

        Returns
        -------
        fiber_length_left : DataFrame
        """
        return self.LeftHem

    def getConnectivityRight(self):
        """
        Returns the connectivity of the right hemisphere of the different
        Desikan Killiany areas.

        Returns
        -------
        connectivity_right : DataFrame
        """
        return self.ConnectivityRight

    def getConnectivityLeft(self):
        """
        Returns the connectivity of the left hemisphere of the different
        Desikan Killiany areas

        Returns
        -------
        connectivity_left : DataFrame
        """
        return self.ConnectivityLeft

    def dump(self, file_path):
        """
        Save all data as a hdf file
        """
        # TODO keep as hdf?
        self.data.to_hdf(file_path, key='HCP', mode='w')


class VolumesDK():
    """
    Loads area volumes in Desikan Killiany parcellation. Provided
    by Alex Goulas.

    Parameters
    ----------
    file_path : string
        Path to data file
    """

    def __init__(self, file_path):
        self.file_path = file_path

        # Read in data
        self.data = loadmat(self.file_path)

        # Preprocess names: Convert to 1d list & strip leading 'ctx-lh-'
        self.names = self.data['HumanDKNames']
        self.names = np.concatenate(self.names.flatten()).tolist()
        self.data = pd.Series(
            data=self.data['Volume'].flatten(),
            index=self.names,
            dtype=np.float64
        )

    def getVolume(self):
        """
        Returns the volume of all areas in the Desikan Killiany
        parcellation. The volume is cubic mm.

        Returns
        -------
        volume : Series
            Volume of the areas
        """
        return self.data


if __name__ == "__main__":

    dk_path = 'experimental_data/hcp_dti/Connectivity_Distances_HCP_DesikanKilliany.mat'
    dk = HcpDesikanKilliany(dk_path)
    print(dk)
    # dk.dump('dk_tmp.hdf')
    print(dk.data['Connectivity (right) [NOS]', 'middletemporal'])

    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(figsize=(10, 10))
    # sns.heatmap(
    #    dk.data['Connectivity (right) [NOS]'],
    #    ax=ax,
    #    xticklabels=1,
    #    yticklabels=1,
    #    cmap='Blues'
    # )
    # plt.tight_layout()
    # plt.show()

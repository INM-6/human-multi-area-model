# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from .. import data_loader as dl

# TODO __repr__(self)

__all__ = [
    "NeuronNumbers",
]

class NeuronNumbers():
    """
    Provides neuron numbers.

    Parameters
    ----------
    surface_area : float
        Surface area of microcolumn.
    source : string
        Source for cytoarchitecture data.
        Implemented: VonEconomoKoskinas
    src_path : string
        Path to cytoarchitecture data.
    target : string
        Target atlas.
        Implemented: From VonEconomoKoskinas to DesikanKilliany
    ei_ratio_path : string
        Path to layer-resolved data for fraction of excitatory neurons.
    min_neurons_per_layer : int
        Minimal number of neurons per layer.
    remove_smaller_layerI : bool
        Remove layers with fewer neurons than layer I.
    """

    def __init__(self, surface_area, source, src_path, ei_ratio_path,
                 min_neurons_per_layer, remove_smaller_layerI, target=None):
        # Collect all parameters, e.g. for later export
        self.params = {
            'surface_area': surface_area,
            'source': source,
            'src_path': src_path,
            'ei_ratio_path': ei_ratio_path,
            'min_neurons_per_layer': min_neurons_per_layer,
            'remove_smaller_layerI': remove_smaller_layerI,
            'target': target,
        }

        self.surface_area = surface_area
        self.layer_list = ['II/III', 'IV', 'V', 'VI']
        self.layer_list_plus1 = ['I'] + self.layer_list
        fraction_E_neurons = dl.ratio_exc_to_inh.ratio_exc_to_inh(ei_ratio_path)
        self.setPopulationDefault(fraction_E_neurons=fraction_E_neurons)

        if source == 'VonEconomoKoskinas':
            # Load VEK data
            vek = dl.voneconomokoskinas.VonEconomoKoskinas(src_path)
            self.area_list = vek.getAreaList()
            NeuronDensities = vek.getNeuronDensity()
            Layerthickness = vek.getLayerthickness()

            # Dict that defines mapping of layers for VEK
            layer_dict = {
                'I': ['I', 'Ia', 'Ib'],
                'II/III': ['II', 'II+III', 'III', 'III(II)',
                           'III(IV)', 'IIIa', 'IIIa/b',
                           'IIIb', 'IIIb/c', 'IIIc'],
                'IV': ['IV', 'IVa', 'IVb', 'IVc'],
                'V': ['V', 'V+VI', 'Va',  'Va1', 'Va2',  'Vb'],
                'VI': ['VI', 'VIa', 'VIa1',  'VIa2',  'VIb']
            }

            # Initialize empty Series for density and thickness
            self.density = pd.Series(
                data=0,
                index=pd.MultiIndex.from_product(
                    [self.area_list, self.layer_list, self.population_list],
                    names=['area', 'layer', 'population']
                ),
                dtype=np.float64
            )
            self.thickness = pd.Series(
                data=0,
                index=pd.MultiIndex.from_product(
                    [self.area_list, self.layer_list_plus1],
                    names=['area', 'layer']
                ),
                dtype=np.float64
            )
            self.layerI_neurons = pd.Series(
                data=0, index=self.area_list, dtype=np.float64
            )

            # Map vEK layers to layer_list using layer_dict
            for area in self.area_list:
                for layer in self.layer_list:
                    layer_vEK = NeuronDensities[area].index.intersection(
                        layer_dict[layer]
                    ).drop_duplicates()
                    if len(layer_vEK) > 0:
                        density_vek = NeuronDensities[area].loc[layer_vEK]
                        thickness_rel = Layerthickness[area].loc[layer_vEK]
                        thickness_rel /= thickness_rel.sum()
                        self.density.loc[area, layer] = (
                            self.frac_pop[layer] * np.sum(
                                density_vek * thickness_rel
                            )
                        ).values
                    # Not in vEK data -> keep default value 0
                    else:
                        self.density.loc[area, layer] = 0
                for layer in self.layer_list_plus1:
                    layer_vEK = Layerthickness[area].index.intersection(
                        layer_dict[layer]
                    ).drop_duplicates()
                    if len(layer_vEK) > 0:
                        self.thickness.loc[area, layer] = np.sum(
                            Layerthickness[area].loc[layer_vEK]
                        )
                    # Not in vEK data -> keep default value 0
                    else:
                        self.thickness.loc[area, layer] = 0
                layer_vEK = NeuronDensities[area].index.intersection(
                    layer_dict['I']
                ).drop_duplicates()
                if len(layer_vEK) > 0:
                    self.layerI_neurons.loc[area] = np.sum(
                        Layerthickness[area].loc[layer_vEK] *
                        NeuronDensities[area].loc[layer_vEK] *
                        self.surface_area
                    )
                # Not in vEK data -> keep default value 0
                else:
                    self.layerI_neurons.loc[area] = 0

            # Map vEK areas to `target` if specified
            if target:
                self.map_atlas(source, target)
        else:
            raise NotImplementedError('Source {} unknown.'.format(source))

        # Assert no NANs
        assert(not self.thickness.isnull().values.any())
        assert(not self.density.isnull().values.any())
        assert(not self.layerI_neurons.isnull().values.any())

        # eliminate layers with less neurons than layer I if
        # remove_smaller_layerI is True; eliminate layers with fewer neurons
        # than min_neurons_per_layer
        neuron_sum = self.getNeuronNumbers().groupby(['area', 'layer']).sum()
        for (area, layer), neuronnumber in neuron_sum.items():
            nn_layerI = self.layerI_neurons.loc[area]
            if remove_smaller_layerI and 0 < neuronnumber < nn_layerI:
                print(f'Dropping layer {layer} in {area} because it '
                      'contains less neurons than layer I.')
                self.thickness.loc[area, layer] = 0
                self.density.loc[area, layer, :] = 0
            elif 0 < neuronnumber < min_neurons_per_layer:
                print(f'Dropping layer {layer} in {area} because it '
                      f'contains {neuronnumber} < {min_neurons_per_layer} '
                      'neurons.')
                self.thickness.loc[area, layer] = 0
                self.density.loc[area, layer, :] = 0

    def map_atlas(self, source, target):
        """
        Map density and thickness from `source` parcellation to `target`
        parcellation.
        """
        if source == 'VonEconomoKoskinas' and target == 'DesikanKilliany':
            # Load the mapping from desikan killiany to von Economo Koskinas
            map_dk_to_vek = dl.mapping_DK_vEK.dk_to_vEK
            # Define the desikan killiany area list,
            # the layer list as well as the data frame that
            # holds the data.
            self.area_list = np.array(list(map_dk_to_vek.keys()))
            density_new = pd.Series(
                data=0,
                index=pd.MultiIndex.from_product(
                    [self.area_list, self.layer_list, self.population_list],
                    names=['area', 'layer', 'population']
                ),
                dtype=np.float64
            )
            thickness_new = pd.Series(
                data=0,
                index=pd.MultiIndex.from_product(
                    [self.area_list, self.layer_list_plus1],
                    names=['area', 'layer']
                ),
                dtype=np.float64
            )
            layerI_neurons_new = pd.Series(
                data=0, index=self.area_list, dtype=np.float64
            )

            # Map from von Economo to Desikan Killiany
            # Here, we assume equal surface area for all vEK areas
            for dk_area in self.area_list:
                # average the thickness over all aggregated ares
                thickness_new.loc[dk_area] = self.thickness.loc[
                    map_dk_to_vek[dk_area]
                ].groupby(level='layer').mean().values
                # weigthed average of densities, weight is given by
                # the relative thicknesses of the layers in the respective
                # areas
                density_dk_area = self.density.loc[map_dk_to_vek[dk_area]]
                density_weight = pd.Series(
                    data=np.repeat(
                        self.thickness.loc[
                            map_dk_to_vek[dk_area], self.layer_list
                        ].values,
                        len(self.population_list)
                    ),
                    index=density_dk_area.index,
                    dtype=np.float64
                )
                density_weight /= np.tile(density_weight.groupby(
                    level=['layer', 'population']).sum().values,
                    len(density_weight.index.unique(level='area'))
                )
                density_weight = density_weight.fillna(0)  # absent layers
                density_new.loc[dk_area] = (
                    density_dk_area * density_weight
                ).groupby(level=['layer', 'population']).sum().values
                # average layer 1 neuron number over all aggregated ares
                layerI_neurons_new.loc[dk_area] = self.layerI_neurons.loc[
                    map_dk_to_vek[dk_area]
                ].mean()
            self.density = density_new
            self.thickness = thickness_new
            self.layerI_neurons = layerI_neurons_new
        else:
            raise NotImplementedError(
                'No mapping from {0} to {1} available.'.format(source, target)
            )

    def setPopulationDefault(self, fraction_E_neurons):
        """
        Sets population_list to ['E', 'I'] and the fraction of E to I.
        """
        self.population_list = ['E', 'I']
        # Fraction of E / I neurons (from Potjans & Diesmann 2014)
        self.frac_pop = pd.Series(
            data=0,
            index=pd.MultiIndex.from_product(
                [self.layer_list, self.population_list],
                names=['layer', 'population']
            ),
            dtype=np.float64
        )
        for layer in self.layer_list:
            self.frac_pop.loc[layer, 'E'] = fraction_E_neurons.loc[layer]
            self.frac_pop.loc[layer, 'I'] = 1. - fraction_E_neurons.loc[layer]

    def getThickness(self):
        """
        Returns the layer thickness in the layers for all areas.

        Returns
        -------
        thickness : Series
        """
        return self.thickness

    def getDensity(self):
        """
        Returns the neuron density in the layers for all areas.

        Returns
        -------
        density : Series
        """
        return self.density

    def getNeuronNumbers(self):
        """
        Returns the neuron number in the layers for all areas.

        Returns
        -------
        neuron_numbers : Series
        """
        # Thickness is only layer resolved and includes layer 1
        thickness_repeat = pd.Series(
            data=np.repeat(
                self.thickness.loc[self.area_list, self.layer_list].values,
                len(self.population_list)
            ),
            index=pd.MultiIndex.from_product(
                [self.area_list, self.layer_list, self.population_list],
                names=['area', 'layer', 'population']
            ),
            dtype=np.float64
        )
        return np.round(
            self.density * thickness_repeat * self.surface_area
        ).astype(np.int64)

    def getTotalThickness(self):
        """
        Returns the total thickness for all areas.

        Returns
        -------
        total_thickness : Series
        """
        return self.thickness.groupby(level='area').sum()

    def getMeanDensity(self):
        """
        Returns the mean neuron density for all areas.

        Returns
        -------
        mean_density : Series
        """
        neuron_number = self.getNeuronNumbers().groupby(level='area').sum()
        total_thickness = self.getTotalThickness()
        return neuron_number / total_thickness / self.surface_area

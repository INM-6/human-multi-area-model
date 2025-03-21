# -*- coding: utf-8 -*-

import numpy as np
import scipy as sci
import pandas as pd

from .. import data_loader as dl

__all__ = [
    "SynapseNumbers",
]

class SynapseNumbers():
    """
    Class that provides population specific synapse numbers.

    Parameters
    ----------
    connectivity : string
        Specifies the connectivity data.
        Implemented: HcpDesikanKilliany
    NN : NeuronNumbers
        An instantiation of the NeuronNumbers class that
        provides infromation about the cytoarchitecture.
    con_path : string
        Path to the connectivity data
    vol_path : string
        Path to the area volume data
    FLN : float
        Fraction of recurrent (intra-area) connections.
    rho_syn : float
        Synapses per cubic mm.
    Z_i : float
        Relative number of cortico-cortical feedback synapses targeting
        excitatory neurons.
    SLN_FF : float
        Determines synaptic target pattern (compare Schmidt et al. 2018).
    SLN_FB : float
        Determines synaptic target pattern (compare Schmidt et al. 2018).
    lmbda : float
        Spatial connectivity decay parameter.
    a0 : float
        Fit parameter for SLN fit from neuron densities.
    a1 : float
        Fit parameter for SLN fit from neuron densities.
    """
    def __init__(self, connectivity, NN, con_path, vol_path, FLN, rho_syn,
                 Z_i, SLN_FF, SLN_FB, lmbda, a0, a1):

        self.NN = NN
        # Collect all parameters, e.g. for later export
        self.params = {
            'connectivity': connectivity,
            'con_path': con_path,
            'vol_path': vol_path,
            'FLN': FLN,
            'rho_syn': rho_syn,
            'Z_i': Z_i,
            'SLN_FF': SLN_FF,
            'SLN_FB': SLN_FB,
            'lmbda': lmbda,
            'a0': a0,
            'a1': a1
        }
        # Get lists from cytoarchitecture data
        self.area_list = NN.area_list
        self.layer_list = NN.layer_list
        self.layer_list_plus1 = NN.layer_list_plus1
        self.population_list = NN.population_list

        # Calculate number of synapses per area (microcircuit)
        local_synapses = rho_syn * NN.surface_area * NN.getTotalThickness()

        # Calculate area surface, area distances and area specific synapse
        # numbers dependent on the connectivity data.
        if connectivity == 'HcpDesikanKilliany':
            # Load area volumes in mm^3
            area_volumes = dl.hcp_dti.VolumesDK(vol_path).getVolume()
            # Assert same atlas
            assert(np.array_equal(area_volumes.index.values, self.area_list))
            # Calculate area specific surface area
            self.area_surface = area_volumes / self.NN.getTotalThickness()

            # Load NOS, randomly take right hemisphere
            NOS = dl.hcp_dti.HcpDesikanKilliany(con_path).getConnectivityRight()
            # Assert same atlas
            assert(np.array_equal(NOS.index.values, self.area_list))
            # Calculate relNOS and account for FLN (cortico-cortical)
            relSynapses = (1 - FLN) * NOS.div(NOS.sum(axis=1), axis=0)
            # Account for FLN (intra-area)
            np.fill_diagonal(relSynapses.values, FLN)
            # Calculate area specific synapse numbers
            self.N_syn_CC = relSynapses.mul(local_synapses, axis=0)
            assert(np.allclose(local_synapses, self.N_syn_CC.sum(axis=1)))
            self.N_syn_CC = np.round(self.N_syn_CC).astype(np.int64)

            # Load atlas specific distances from fiber length.
            # Randomly take right hemisphere.
            self.dist = dl.hcp_dti.HcpDesikanKilliany(con_path).getFiberLengthRight()
        else:
            raise NotImplementedError(
                "Connectivity {} unknown.".format(connectivity)
            )
        self.NOS = NOS

        # Calculate SLN values
        meanDensity = NN.getMeanDensity()
        self.SLN = self.calcSLN(meanDensity, meanDensity, a0, a1)

        # Calculate hierarchical directionality based on SLN
        self.directionality = self.calcDirectionality(SLN_FF, SLN_FB)

        # Calculate layer specific connectivity
        self.N_syn, self.N_syn_ext = self.populationSpecificSynapseNumbers(
            SLN_FF, SLN_FB, Z_i, lmbda
        )
        assert(np.allclose(
            local_synapses,
            self.N_syn.sum(axis=1).groupby('area').sum() +
            self.N_syn_ext.groupby('area').sum()
        ))

        # Assert no NANs
        assert(not self.dist.isnull().values.any())
        assert(not self.N_syn_CC.isnull().values.any())
        assert(not self.N_syn.isnull().values.any())
        assert(not self.N_syn_ext.isnull().values.any())

    def getDistance(self):
        """
        Returns area distances.

        Returns
        -------
        dist : DataFrame
        """
        return self.dist

    def getAreaLevelSynapseNumbers(self):
        """
        Returns area resolved synapse numbers.

        Returns
        -------
        synapse_numbers_area : DataFrame
        """
        return self.N_syn_CC

    def getSynapseNumbers(self):
        """
        Returns population resolved synapse numbers.

        Returns
        -------
        synapse_numbers_external : DataFrame
        """
        return self.N_syn

    def getSynapseNumbersExternal(self):
        """
        Returns population resolved external synapse numbers.

        Returns
        -------
        synapse_numbers_external : DataFrame
        """
        return self.N_syn_ext

    def getSLN(self):
        """
        Returns fitted SLN values.

        Returns
        -------
        SLN : DataFrame
        """
        return self.SLN

    @staticmethod
    def calcSLN(rhoTarget, rhoSource, a0, a1):
        r"""
        Fit parameters a0, a1 from Schmidt et al. (2018), Fig. 5

        .. math::
           \mathrm{SLN} = \mathrm{SNCDF}[a_0 + a_1 \log(
              \\rho_{\mathrm{target}} / \\rho_{\mathrm{source}} )]

        where SNCDF is the cumulative density function of the standard
        normal distribution.

        Returns
        -------
        SLN : DataFrame
            Area specific SLN value
        """
        logratio = np.log(np.outer(rhoTarget, 1./rhoSource))
        SLN = pd.DataFrame(
            data=sci.stats.norm.cdf(a0 + a1*logratio, loc=0, scale=1),
            index=rhoTarget.index.values,
            columns=rhoSource.index.values
        )
        return SLN

    def calcDirectionality(self, SLN_FF, SLN_FB):
        """
        Calculates the hierarchical directionality of a connection (Whether
        it's feedback, feedforward, or lateral) and places a string
        (FB, FF, lat) at the corresponding place in the dataframe.

        Returns
        -------
        d : DataFrame
            Area specific hierarchical directionality
        """
        def tmp(val, SLN_FF, SLN_FB):
            if val < SLN_FB:
                return 'FB'
            elif SLN_FB <= val < SLN_FF:
                return 'lat'
            else:
                return 'FF'
        d = self.SLN.map(lambda x: tmp(x, SLN_FF, SLN_FB))
        return d

    def populationSpecificSynapseNumbers(self, SLN_FF, SLN_FB, Z_i, lmbda):
        """
        Breaks area specific synapse numbers down to populations, i.e.
        into type I, II and III.

        Returns
        -------
        N_syn : DataFrame
            Population resolved internal syapse numbers.
        N_syn_ext : DataFrame
            Population resolved internal syapse numbers.
        """
        from itertools import product

        # Datastrucutres for cortico-cortical connectivity
        multiindex = pd.MultiIndex.from_product(
            [self.area_list, self.layer_list, self.population_list],
            names=['area', 'layer', 'population']
        )
        multiindex_noarea = pd.MultiIndex.from_product(
            [self.layer_list, self.population_list],
            names=['layer', 'population']
        )
        N_syn = pd.DataFrame(
            data=0,
            index=multiindex,
            columns=multiindex
        ).sort_index()
        N_syn_ext = pd.Series(
            data=0,
            index=multiindex
        )
        X = pd.Series(
            data=0.,
            index=multiindex_noarea
        )
        Y = pd.Series(
            data=0.,
            index=self.layer_list_plus1
        )

        binzegger = dl.synapse_cellbody_probability.binzegger
        # Fraction of excitatory and inhibitory connections onto layers II/III,
        # IV, V, and VI. Taken from binzegger fractions
        E_connections_fraction = binzegger.loc[
                pd.IndexSlice[:, 'E'], :
                ].sum(axis=0)
        I_connections_fraction = binzegger.loc[
                pd.IndexSlice[:, 'I'], :
                ].sum(axis=0)

        mohan = dl.synapse_cellbody_probability.mohan
        # Multiply the mohan data, which has only information on E connections
        # (all I entries are 0), with the fraction of E connections we want to
        # achieve. This fraction is taken from the binzegger data. This only
        # adjusts the excitatory connections.
        assert np.allclose(mohan.loc[pd.IndexSlice[:, 'I'], :].values, 0.)
        mohan_adjusted = mohan * E_connections_fraction

        # Loop over all connections onto inhibitory I neurons and assign the
        # correct fraction to those connections. By keeping the layer l fixed
        # we introduce the assumption that all inhibitory connections are local
        # (stay in the area).
        mohan_onto_I = mohan_adjusted.loc[pd.IndexSlice[:, 'I'], :]
        for (l, p), _ in mohan_onto_I.iterrows():
            mohan_adjusted.loc[(l, p), l] = I_connections_fraction[l]

        total_thickness = self.NN.getTotalThickness()

        for areaSource, areaTarget in product(self.area_list, self.area_list):
            N_syn_loc = self.N_syn_CC.loc[areaTarget, areaSource]

            # type I or II
            if areaSource == areaTarget:
                NN_loc = self.NN.getNeuronNumbers().loc[areaTarget]
                total_thick_loc = total_thickness.loc[areaTarget]
                surface_loc = self.area_surface.loc[areaTarget]

                # split into type I and II
                # epsrel: 1e-1 is sufficiently accurate, testing against
                # different parameters lead to the same results. Higher
                # accuracy takes too long.
                # TODO improve this!
                P_in = sci.integrate.nquad(
                    self.integrand_connectivity_profile_exp,
                    ranges=[
                        [0, 1e3/np.sqrt(np.pi)],
                        [0, 1e3/np.sqrt(np.pi)],
                        [0, 2*np.pi],
                        [0, total_thick_loc]
                    ],
                    args=[2*np.pi, total_thick_loc, lmbda],
                    opts={'epsrel': 1e-1}
                )[0]
                P_out = sci.integrate.nquad(
                    self.integrand_connectivity_profile_exp,
                    ranges=[
                        [1e3/np.sqrt(np.pi), 1e3*np.sqrt(surface_loc/np.pi)],
                        [0, 1e3/np.sqrt(np.pi)],
                        [0, 2*np.pi],
                        [0, total_thick_loc]
                    ],
                    args=[2*np.pi, total_thick_loc, lmbda],
                    opts={'epsrel': 1e-1}
                )[0]
                N_syn_loc_I = P_in / (P_in + P_out) * N_syn_loc
                N_syn_loc_II = P_out / (P_in + P_out) * N_syn_loc

                # break type I synapses down to population level
                rel_p_PD = dl.microcircuit.p.mul(NN_loc, axis=0).mul(NN_loc, axis=1)
                rel_p_PD /= rel_p_PD.values.sum()
                N_syn.loc[(areaTarget), (areaSource)] = (
                    np.round(rel_p_PD.values * N_syn_loc_I)
                )

                # break type II synapses down to population level
                rel_Indeg_ext = dl.microcircuit.K_ext * NN_loc
                rel_Indeg_ext /= rel_Indeg_ext.values.sum()
                N_syn_ext.loc[areaTarget] = (
                    np.round(rel_Indeg_ext.values * N_syn_loc_II)
                )

            # type III synapses
            else:
                if N_syn_loc > 0.:
                    thickSource = self.NN.getThickness().loc[areaSource]
                    thickTarget = self.NN.getThickness().loc[areaTarget]
                    densSource = self.NN.getDensity().loc[areaSource]
                    nnTarget = self.NN.getNeuronNumbers().loc[areaTarget]
                    SLN_loc = self.SLN.loc[areaTarget, areaSource]

                    # Whether connection is FF, FB, lateral
                    hierarchical_direction = self.directionality.loc[
                        areaTarget, areaSource
                    ]

                    # Create X vector from eq. (3) Schmidt et al. SuppMat
                    X.loc['II/III', 'E'] = SLN_loc
                    dens5E = densSource.loc['V', 'E']
                    thick5 = thickSource.loc['V']
                    dens6E = densSource.loc['VI', 'E']
                    thick6 = thickSource.loc['VI']
                    ratio5E6E = dens5E*thick5 / (dens5E*thick5 + dens6E*thick6)
                    X.loc['V', 'E'] = (1. - SLN_loc) * ratio5E6E
                    X.loc['VI', 'E'] = (1. - SLN_loc) * (1. - ratio5E6E)

                    # Create synapse target pattern
                    if SLN_loc > SLN_FF:
                        if thickTarget.loc['IV'] > 0:
                            P_t = ['IV']
                        else:
                            # personal comm. CH -> Beul et al. 2015
                            P_t = ['II/III']
                    elif SLN_loc < SLN_FB:
                        P_t = ['I', 'II/III', 'V', 'VI']
                    else:
                        P_t = ['I', 'II/III', 'IV', 'V', 'VI']
                    if np.allclose(thickTarget.loc[P_t].sum(), 0):
                        raise NotImplementedError(
                          f'Impossible to assign target pattern {P_t} for '
                          f'{areaTarget} with thickness \n{thickTarget}'
                        )

                    # Create Y vector from eq. (3) Schmidt et al. SuppMat
                    Y *= 0.
                    relThickTargetPattern = (
                        thickTarget.loc[P_t] / thickTarget.loc[P_t].sum()
                    )
                    Y.loc[P_t] = relThickTargetPattern

                    # calculate sum_v Y_v P(i|s_cc \in v)
                    target_prob = (Y * mohan_adjusted).sum(axis=1)

                    # set target probability to zero for populations without
                    # neurons; redistribute probability equality among the
                    # remaining populations
                    target_prob[nnTarget == 0] = 0.
                    target_prob = target_prob / target_prob.sum()

                    N_syn_loc_tmp = pd.DataFrame(
                            N_syn_loc * np.outer(target_prob, X),
                            index=multiindex_noarea,
                            columns=multiindex_noarea
                            )

                    if hierarchical_direction == 'FB':
                        # If the connection goes into FeedBack direction we
                        # make sure that Z_i (usually 93%) of the connections
                        # are onto excitatory neurons and 1 - Z_i connections
                        # go onto inhibitory neurons. This resembles the fact
                        # that FB projections preferentially target excitatory
                        # neurons. See also multi-area model code and paper.

                        # 1st: Determine number of synapses onto E and I
                        # neurons
                        E_sum = N_syn_loc_tmp.loc[
                            pd.IndexSlice[:, 'E'], :
                        ].sum().sum()
                        I_sum = N_syn_loc_tmp.loc[
                            pd.IndexSlice[:, 'I'], :
                        ].sum().sum()

                        # 2nd: Determine fraction of E and I connections
                        alpha_E = E_sum / (E_sum + I_sum)
                        alpha_I = I_sum / (E_sum + I_sum)

                        # 3rd: Scale E and I connections with the corresponding
                        # factors to achieve Z_i (or 1-Z_i)
                        N_syn_loc_tmp.loc[
                            pd.IndexSlice[:, 'E'], :
                        ] *= Z_i / alpha_E
                        N_syn_loc_tmp.loc[
                            pd.IndexSlice[:, 'I'], :
                        ] *= (1 - Z_i) / alpha_I

                    N_syn.loc[
                        (areaTarget), (areaSource)
                    ] = N_syn_loc_tmp.round().values

        # Cast to integer
        N_syn = np.round(N_syn).astype(np.int64)
        N_syn_ext = np.round(N_syn_ext).astype(np.int64)
        return N_syn, N_syn_ext

    @staticmethod
    def integrand_connectivity_profile_exp(r1, r2, phi, z, phi_max, z_max,
                                           lmbda):
        """
        Returns an exponential probability density with decay parameter
        lambda. All units in micron, using cylinder coordinates.

        Parameters
        ----------
        r1, r2 : float
            Radial coordinate of the neurons
        phi : float
            Relative angle betweeen the neurons
        z : float
            Relative distance in z-direction
        phi_max : float
            Upper bound for phi
        z_max : float
            Upper bound for z
        lmbda : float
            Decay parameter
        """
        dist = np.sqrt(r1**2 - 2*r1*r2*np.cos(phi) + r2**2 + z**2)
        return 4 * r1*r2 * (phi_max - phi) * (z_max - z) * np.exp(-dist/lmbda)

from .. import get_backend, default_backend, events
from ..interpolate import _hfinterpolator_code0 ,_hfinterpolator_code1

class normsys_combinedmod(object):
    def __init__(self, normsys_mods, pdfconfig, mega_mods):
        self._parindices = list(range(len(pdfconfig.suggested_init())))
        self._normsys_indices = [self._parindices[pdfconfig.par_slice(m)] for m in normsys_mods]
        self._normsys_histoset = [
            [
                [
                    mega_mods[s][m]['data']['lo'],
                    mega_mods[s][m]['data']['nom_data'],
                    mega_mods[s][m]['data']['hi'],
                ]
                for s in pdfconfig.samples
            ] for m in normsys_mods
        ]
        self._normsys_mask = [
            [
                [
                    mega_mods[s][m]['data']['mask'],
                ]
                for s in pdfconfig.samples
            ] for m in normsys_mods
        ]

        if len(normsys_mods):
            self.interpolator = _hfinterpolator_code1(self._normsys_histoset)

        self._precompute()
        events.subscribe('tensorlib_changed')(self._precompute)

    def _precompute(self):
        tensorlib, _ = get_backend()
        self.normsys_mask = tensorlib.astensor(self._normsys_mask)
        self.normsys_default = tensorlib.ones(self.normsys_mask.shape)
        self.normsys_indices = tensorlib.astensor(self._normsys_indices, dtype='int')

    def apply(self,pars):
        tensorlib, _ = get_backend()
        if not tensorlib.shape(self.normsys_indices)[0]:
            return
        normsys_alphaset = tensorlib.gather(pars,self.normsys_indices)
        results_norm   = self.interpolator(normsys_alphaset)

        #either rely on numerical no-op or force with line below
        results_norm   = tensorlib.where(self.normsys_mask,results_norm,self.normsys_default)
        return results_norm


class histosys_combinedmod(object):
    def __init__(self,histosys_mods,pdfconfig,mega_mods):
        self._parindices = list(range(len(pdfconfig.suggested_init())))
        self._histo_indices = [self._parindices[pdfconfig.par_slice(m)] for m in histosys_mods]
        self._histosys_histoset = [
            [
                [
                    mega_mods[s][m]['data']['lo_data'],
                    mega_mods[s][m]['data']['nom_data'],
                    mega_mods[s][m]['data']['hi_data'],
                ]
                for s in pdfconfig.samples
            ] for m in histosys_mods
        ]
        self._histosys_mask = [
            [
                [
                    mega_mods[s][m]['data']['mask'],
                ]
                for s in pdfconfig.samples
            ] for m in histosys_mods
        ]

        if len(histosys_mods):
            self.interpolator = _hfinterpolator_code0(self._histosys_histoset)

        self._precompute()
        events.subscribe('tensorlib_changed')(self._precompute)

    def _precompute(self):
        tensorlib, _ = get_backend()
        self.histosys_mask = tensorlib.astensor(self._histosys_mask)
        self.histosys_default = tensorlib.zeros(self.histosys_mask.shape)
        self.histo_indices = tensorlib.astensor(self._histo_indices, dtype='int')

    def apply(self,pars):
        tensorlib, _ = get_backend()
        if not tensorlib.shape(self.histo_indices)[0]:
            return
        histosys_alphaset = tensorlib.gather(pars,self.histo_indices)
        results_histo   = self.interpolator(histosys_alphaset)
        # either rely on numerical no-op or force with line below
        results_histo   = tensorlib.where(self.histosys_mask,results_histo,self.histosys_default)
        return results_histo

class normfac_combinedmod(object):
    def __init__(self,normfac_mods,pdfconfig,mega_mods):
        self._parindices = list(range(len(pdfconfig.suggested_init())))
        self._normfac_indices = [self._parindices[pdfconfig.par_slice(m)] for m in normfac_mods]
        self._normfactor_mask = [
            [
                [
                    mega_mods[s][m]['data']['mask'],
                ]
                for s in pdfconfig.samples
            ] for m in normfac_mods
        ]

        self._precompute()
        events.subscribe('tensorlib_changed')(self._precompute)

    def _precompute(self):
        tensorlib, _ = get_backend()
        self.normfactor_mask = default_backend.astensor(self._normfactor_mask)
        self.normfactor_default = default_backend.ones(self.normfactor_mask.shape)
        self.normfac_indices = default_backend.astensor(self._normfac_indices, dtype='int')

    def apply(self,pars):
        tensorlib, _ = get_backend()
        normfac_indices = tensorlib.astensor(self.normfac_indices, dtype='int')
        normfac_mask = tensorlib.astensor(self.normfactor_mask)
        if not tensorlib.shape(normfac_indices)[0]:
            return
        normfactors = tensorlib.gather(pars,normfac_indices)
        results_normfac = normfac_mask * tensorlib.reshape(normfactors,tensorlib.shape(normfactors) + (1,1))
        results_normfac = tensorlib.where(normfac_mask,results_normfac,tensorlib.astensor(self.normfactor_default))
        return results_normfac

class staterror_combined(object):
    def __init__(self,staterr_mods,pdfconfig,mega_mods):
        self._parindices = list(range(len(pdfconfig.suggested_init())))
        self._staterror_indices = [self._parindices[pdfconfig.par_slice(m)] for m in staterr_mods]
        self._staterr_mods = staterr_mods
        self._staterror_mask = [
            [
                [
                    mega_mods[s][m]['data']['mask'],
                ]
                for s in pdfconfig.samples
            ] for m in staterr_mods
        ]
        self.__staterror_uncrt = default_backend.astensor([
            [
                [
                    mega_mods[s][m]['data']['uncrt'],
                    mega_mods[s][m]['data']['nom_data'],
                ]
                for s in pdfconfig.samples
            ] for m in staterr_mods
        ])

        if self._staterror_indices:
            access_rows = []
            staterror_mask = default_backend.astensor(self._staterror_mask)
            for mask,inds in zip(staterror_mask, self._staterror_indices):
                summed_mask = default_backend.sum(mask[:,0,:],axis=0)
                assert default_backend.shape(summed_mask[summed_mask >  0]) == default_backend.shape(default_backend.astensor(inds))
                summed_mask[summed_mask >  0] = inds
                summed_mask[summed_mask == 0] = -1
                access_rows.append(summed_mask.tolist())
            self._factor_access_indices = default_backend.tolist(default_backend.stack(access_rows))
            self.finalize(pdfconfig)
        else:
            self._factor_access_indices = None

        self._precompute()
        events.subscribe('tensorlib_changed')(self._precompute)

    def _precompute(self):
        tensorlib, _ = get_backend()
        self.staterror_mask = tensorlib.astensor(self._staterror_mask)
        self.staterror_default = tensorlib.ones(tensorlib.shape(self.staterror_mask))

        if self._staterror_indices:
            self.factor_access_indices = tensorlib.astensor(self._factor_access_indices, dtype='int')
            self.default_value = tensorlib.astensor([1.0])
            self.sample_ones   = tensorlib.ones(tensorlib.shape(self.staterror_mask)[1])
            self.alpha_ones    = tensorlib.astensor([1])
        else:
            self.factor_access_indices = None

    def finalize(self,pdfconfig):
        staterror_mask = default_backend.astensor(self._staterror_mask)
        for this_mask, uncert_this_mod,mod in zip(staterror_mask, self.__staterror_uncrt, self._staterr_mods):
            active_nominals = default_backend.where(
                this_mask[:,0,:], uncert_this_mod[:,1,:],
                default_backend.zeros(uncert_this_mod[:,1,:].shape)
            )
            summed_nominals = default_backend.sum(active_nominals, axis = 0)

            # the below tries to filter cases in which this modifier is not
            # used by checking non zeroness.. should probably use mask
            numerator   = default_backend.where(
                uncert_this_mod[:,1,:] > 0,
                uncert_this_mod[:,0,:],
                default_backend.zeros(uncert_this_mod[:,1,:].shape)
            )
            denominator = default_backend.where(
                summed_nominals > 0,
                summed_nominals,
                default_backend.ones(uncert_this_mod[:,1,:].shape)
            )
            relerrs = numerator/denominator
            sigmas = default_backend.sqrt(
                default_backend.sum(
                    default_backend.power(relerrs,2),axis=0
                )
            )
            assert len(sigmas[sigmas>0]) == pdfconfig.param_set(mod).n_parameters
            pdfconfig.param_set(mod).sigmas = default_backend.tolist(sigmas[sigmas>0])

    def apply(self,pars):
        tensorlib, _ = get_backend()
        if self.factor_access_indices is None:
            return
        select_from = tensorlib.concatenate([pars,self.default_value])
        factor_row = tensorlib.gather(select_from, self.factor_access_indices)

        results_staterr = tensorlib.einsum('s,a,mb->msab',
                tensorlib.astensor(self.sample_ones),
                tensorlib.astensor(self.alpha_ones),
                factor_row
        )

        results_staterr = tensorlib.where(
            self.staterror_mask,
            results_staterr,
            self.staterror_default
        )
        return results_staterr

class shapefactor_combined(object):
    def __init__(self,shapefactor_mods,pdfconfig,mega_mods):
        self._parindices = list(range(len(pdfconfig.suggested_init())))
        self._shapefactor_indices = [self._parindices[pdfconfig.par_slice(m)] for m in shapefactor_mods]
        self._shapefactor_mask = [
            [
                [
                    mega_mods[s][m]['data']['mask'],
                ]
                for s in pdfconfig.samples
            ] for m in shapefactor_mods
        ]

        self._precompute()
        events.subscribe('tensorlib_changed')(self._precompute)

    def _precompute(self):
        tensorlib, _ = get_backend()
        self.shapefactor_mask = tensorlib.astensor(self._shapefactor_mask)
        self.shapefactor_default = tensorlib.ones(tensorlib.shape(self.shapefactor_mask))
        self.shapefactor_indices = tensorlib.astensor(self._shapefactor_indices, dtype='int')

    def apply(self,pars):
        tensorlib, _ = get_backend()
        shapefactor_indices = tensorlib.astensor(self.shapefactor_indices, dtype='int')
        shapefactor_mask = tensorlib.astensor(self.shapefactortor_mask)
        if not tensorlib.shape(shapefactor_indices)[0]:
            return
        shapefactors = tensorlib.gather(pars,shapefactor_indices)
        results_shapefactor = shapefactor_mask * tensorlib.reshape(shapefactors,tensorlib.shape(shapefactors) + (1,1))
        results_shapefactor = tensorlib.where(shapefactor_mask,results_shapefactor,tensorlib.astensor(self.shapefactor_default))
        return results_shapefactor

class shapesys_combined(object):
    def __init__(self,shapesys_mods,pdfconfig,mega_mods):
        self._shapesys_mods = shapesys_mods
        self._parindices = list(range(len(pdfconfig.suggested_init())))
        self._shapesys_indices = [self._parindices[pdfconfig.par_slice(m)] for m in shapesys_mods]
        self._shapesys_mask = [
            [
                [
                    mega_mods[s][m]['data']['mask'],
                ]
                for s in pdfconfig.samples
            ] for m in shapesys_mods
        ]
        self.__shapesys_uncrt = default_backend.astensor([
            [
                [
                    mega_mods[s][m]['data']['uncrt'],
                    mega_mods[s][m]['data']['nom_data'],
                ]
                for s in pdfconfig.samples
            ] for m in shapesys_mods
        ])

        if self._shapesys_indices:
            access_rows = []
            shapesys_mask = default_backend.astensor(self._shapesys_mask)
            for mask,inds in zip(shapesys_mask, self._shapesys_indices):
                summed_mask = default_backend.sum(mask[:,0,:],axis=0)
                assert default_backend.shape(summed_mask[summed_mask >  0]) == default_backend.shape(default_backend.astensor(inds))
                summed_mask[summed_mask >  0] = inds
                summed_mask[summed_mask == 0] = -1
                access_rows.append(summed_mask.tolist())
            self._factor_access_indices = default_backend.tolist(default_backend.stack(access_rows))
            self.finalize(pdfconfig)
        else:
            self._factor_access_indices = None

        self._precompute()
        events.subscribe('tensorlib_changed')(self._precompute)

    def _precompute(self):
        tensorlib, _ = get_backend()
        self.shapesys_mask = tensorlib.astensor(self._shapesys_mask)
        self.shapesys_default = tensorlib.ones(tensorlib.shape(self.shapesys_mask))

        if self._shapesys_indices:
            self.factor_access_indices = tensorlib.astensor(self._factor_access_indices, dtype='int')
            self.default_value = tensorlib.astensor([1.0])
            self.sample_ones   = tensorlib.ones(tensorlib.shape(self.shapesys_mask)[1])
            self.alpha_ones    = tensorlib.astensor([1])
        else:
            self.factor_access_indices = None

    def finalize(self,pdfconfig):
        for uncert_this_mod,mod in zip(self.__shapesys_uncrt,self._shapesys_mods):
            unc_nom = default_backend.astensor([x for x in uncert_this_mod[:,:,:] if any(x[0][x[0]>0])])
            unc = unc_nom[0,0]
            nom = unc_nom[0,1]
            unc_sq = default_backend.power(unc,2)
            nom_sq = default_backend.power(nom,2)

            #the below tries to filter cases in which
            #this modifier is not used by checking non
            #zeroness.. shoudl probably use mask
            numerator   = default_backend.where(
                unc_sq > 0,
                nom_sq,
                default_backend.zeros(unc_sq.shape)
            )
            denominator = default_backend.where(
                unc_sq > 0,
                unc_sq,
                default_backend.ones(unc_sq.shape)
            )

            factors = numerator/denominator
            factors = factors[factors>0]
            assert len(factors) == pdfconfig.param_set(mod).n_parameters
            pdfconfig.param_set(mod).factors = default_backend.tolist(factors)
            pdfconfig.param_set(mod).auxdata = default_backend.tolist(factors)

    def apply(self,pars):
        tensorlib, _ = get_backend()
        if self.factor_access_indices is None:
            return
        tensorlib, _ = get_backend()

        factor_row = tensorlib.gather(tensorlib.concatenate([tensorlib.astensor(pars), self.default_value]), self.factor_access_indices)

        results_shapesys = tensorlib.einsum('s,a,mb->msab',
                tensorlib.astensor(self.sample_ones),
                tensorlib.astensor(self.alpha_ones),
                factor_row)

        results_shapesys = tensorlib.where(self.shapesys_mask,results_shapesys,self.shapesys_default)
        return results_shapesys

from .. import get_backend, default_backend
from ..interpolate import _hfinterpolator_code0 ,_hfinterpolator_code1

class normsys_combinedmod(object):
    def __init__(self,normsys_mods,pdfconfig,mega_mods):
        samples = pdfconfig.samples

        tensorlib, _ = get_backend()
        self.parindices = list(range(len(pdfconfig.suggested_init())))
        self.normsys_histoset = [
            [
                [
                    mega_mods[s][m]['data']['lo'],
                    mega_mods[s][m]['data']['nom_data'],
                    mega_mods[s][m]['data']['hi'],
                ]
                for s in samples
            ] for m in normsys_mods
        ]

        self.normsys_mask = tensorlib.astensor([
            [
                [
                    mega_mods[s][m]['data']['mask'],
                ]
                for s in samples
            ] for m in normsys_mods
        ])
        self.normsys_default = tensorlib.ones(self.normsys_mask.shape)


        self.normsys_indices = tensorlib.astensor([
            self.parindices[pdfconfig.par_slice(m)] for m in normsys_mods
        ], dtype='int')

        if len(normsys_mods):
            self.interpolator = _hfinterpolator_code1(self.normsys_histoset)


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
        tensorlib, _ = get_backend()
        samples = pdfconfig.samples

        self.parindices = list(range(len(pdfconfig.suggested_init())))
        self.histosys_histoset = [
            [
                [
                    mega_mods[s][m]['data']['lo_data'],
                    mega_mods[s][m]['data']['nom_data'],
                    mega_mods[s][m]['data']['hi_data'],
                ]
                for s in samples
            ] for m in histosys_mods
        ]

        self.histosys_mask = tensorlib.astensor([
            [
                [
                    mega_mods[s][m]['data']['mask'],
                ]
                for s in samples
            ] for m in histosys_mods
        ])
        self.histosys_default = tensorlib.zeros(self.histosys_mask.shape)

        self.histo_indices = tensorlib.astensor([
            self.parindices[pdfconfig.par_slice(m)] for m in histosys_mods
        ], dtype='int')

        if len(histosys_mods):
            self.interpolator = _hfinterpolator_code0(self.histosys_histoset)

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
        samples = pdfconfig.samples

        self.parindices = list(range(len(pdfconfig.suggested_init())))
        tensorlib, _ = get_backend()
        self.normfactor_mask = tensorlib.astensor([
            [
                [
                    mega_mods[s][m]['data']['mask'],
                ]
                for s in samples
            ] for m in normfac_mods
        ])
        self.normfactor_default = tensorlib.ones(self.normfactor_mask.shape)

        self.normfac_indices = tensorlib.astensor([self.parindices[pdfconfig.par_slice(m)] for m in normfac_mods ], dtype='int')

    def apply(self,pars):
        tensorlib, _ = get_backend()
        if not tensorlib.shape(self.normfac_indices)[0]:
            return
        normfactors = tensorlib.gather(pars,self.normfac_indices)
        results_normfac = self.normfactor_mask * tensorlib.reshape(normfactors,tensorlib.shape(normfactors) + (1,1))
        results_normfac = tensorlib.where(self.normfactor_mask,results_normfac,self.normfactor_default)
        return results_normfac

class staterror_combined(object):
    def __init__(self,staterr_mods,pdfconfig,mega_mods):
        channels = pdfconfig.channels
        samples = pdfconfig.samples
        channel_nbins = pdfconfig.channel_nbins

        start_index = 0
        channel_slices = []
        for c in channels:
            end_index = start_index + channel_nbins[c]
            channel_slices.append(slice(start_index,end_index))
            start_index = end_index

        binindices = list(range(sum(list(channel_nbins.values()))))
        channel_slice_map = {
            c:binindices[sl] for c,sl in zip(channels,channel_slices)
        }

        parindices = list(range(len(pdfconfig.suggested_init())))
        self.parindices = parindices

        self._staterror_mask = default_backend.astensor([
            [
                [
                    mega_mods[s][m]['data']['mask'],
                ]
                for s in samples
            ] for m in staterr_mods
        ])
        self._staterror_default = default_backend.ones(
            default_backend.shape(self._staterror_mask)
        )

        stat_parslices  = [pdfconfig.par_slice(m) for m in staterr_mods]
        self.stat_parslices = stat_parslices
        if stat_parslices:
            tensorlib, _ = get_backend()

            factor_access_indices = default_backend.concatenate([
                    default_backend.where(
                        default_backend.sum(msk,axis=0) > 0,
                        default_backend.ones(msk[0].shape)*self.parindices[sl][0],
                        -default_backend.ones(msk[0].shape)
                )
                for msk,sl in zip(self._staterror_mask,stat_parslices)
            ])

            self.factor_access_indices = tensorlib.astensor(factor_access_indices,dtype='int')
            self.default_value = tensorlib.astensor([1.0])
            self.sample_ones   = tensorlib.ones(len(samples))
            self.alpha_ones    = tensorlib.astensor([1])
            self.staterror_mask = default_backend.astensor(default_backend.tolist(self._staterror_mask))
            self.staterror_default = default_backend.astensor(default_backend.tolist(self._staterror_default))
        else:
            self.factor_access_indices = None


    def apply(self,pars):
        tensorlib, _ = get_backend()
        if self.factor_access_indices is None:
            return
        select_from = tensorlib.concatenate([pars,self.default_value])
        factor_row = tensorlib.gather(
            select_from,self.factor_access_indices
        )

        results_staterr = tensorlib.einsum('s,a,mb->msab',
                self.sample_ones,
                self.alpha_ones,
                factor_row
        )

        results_staterr = tensorlib.where(
            self.staterror_mask,
            results_staterr,
            self.staterror_default
        )
        return results_staterr

class shapesys_combined(object):
    def __init__(self,shapesys_mods,pdfconfig,mega_mods):
        channels = pdfconfig.channels
        samples = pdfconfig.samples
        channel_nbins = pdfconfig.channel_nbins

        start_index = 0
        channel_slices = []
        for c in channels:
            end_index = start_index + channel_nbins[c]
            channel_slices.append(slice(start_index,end_index))
            start_index = end_index

        parindices = list(range(len(pdfconfig.suggested_init())))
        binindices = list(range(sum(list(channel_nbins.values()))))
        channel_slice_map = {
            c:binindices[sl] for c,sl in zip(channels,channel_slices)
        }

        self._shapesys_mask = default_backend.astensor([
            [
                [
                    mega_mods[s][m]['data']['mask'],
                ]
                for s in samples
            ] for m in shapesys_mods
        ])
        self._shapesys_default = default_backend.ones(self._shapesys_mask.shape)

        shapesys_parslices  = [pdfconfig.par_slice(m) for m in shapesys_mods]

        if shapesys_parslices:
            tensorlib, _ = get_backend()

            factor_access_indices = default_backend.concatenate([
                    default_backend.where(
                        default_backend.sum(msk,axis=0) > 0,
                        default_backend.ones(msk[0].shape)*self.parindices[sl][0],
                        -default_backend.ones(msk[0].shape)
                )
                for msk,sl in zip(self._shapesys_mask,shapesys_parslices)
            ])



            self.sample_ones = tensorlib.ones(len(samples))
            self.alpha_ones = tensorlib.astensor([1])
            self.factor_access_indices = tensorlib.astensor(factor_access_indices,dtype='int')
            self.default_value = tensorlib.astensor([1.])
        else:
            self.factor_row_indices = None

    def apply(self,pars):
        tensorlib, _ = get_backend()
        if self.factor_row_indices is None:
            return
        tensorlib, _ = get_backend()
        factor_row = tensorlib.gather(tensorlib.concatenate([pars,self.default_value]),self.factor_row_indices)

        results_shapesys = tensorlib.einsum('s,a,mb->msab',
                self.sample_ones,
                self.alpha_ones,
                factor_row)

        results_shapesys = tensorlib.where(self.shapesys_mask,results_shapesys,self.shapesys_default)
        return results_shapesys

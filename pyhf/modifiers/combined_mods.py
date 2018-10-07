from .. import get_backend, default_backend
from ..interpolate import _hfinterpolator_code0 ,_hfinterpolator_code1

class normsys_combinedmod(object):
    def __init__(self,normsys_mods,pdf):
        pdfconfig = pdf.config
        do_samples = pdf.do_samples
        mega_mods = pdf.mega_mods
        mega_samples = pdf.mega_samples



        tensorlib, _ = get_backend()
        self.parindices = list(range(len(pdfconfig.suggested_init())))
        self.normsys_histoset = [
            [
                [
                    mega_mods[s][m]['data']['lo'],
                    [1.]*len(mega_samples[s]['nom']),
                    mega_mods[s][m]['data']['hi'],
                ]
                for s in do_samples
            ] for m in normsys_mods
        ]

        self.normsys_mask = tensorlib.astensor([
            [
                [
                    mega_mods[s][m]['data']['mask'],
                ]
                for s in do_samples
            ] for m in normsys_mods
        ])
        self.normsys_default = tensorlib.ones(self.normsys_mask.shape)


        self.normsys_indices = tensorlib.astensor([
            self.parindices[pdfconfig.par_slice(m)] for m in normsys_mods
        ], dtype='int')

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
    def __init__(self,histosys_mods,pdf):
        tensorlib, _ = get_backend()
        pdfconfig = pdf.config
        mega_mods = pdf.mega_mods
        mega_samples = pdf.mega_samples
        do_samples = pdf.do_samples

        self.parindices = list(range(len(pdfconfig.suggested_init())))
        self.histosys_histoset = [
            [
                [
                    mega_mods[s][m]['data']['lo_data'],
                    mega_samples[s]['nom'],
                    mega_mods[s][m]['data']['hi_data'],
                ]
                for s in do_samples
            ] for m in histosys_mods
        ]

        self.histosys_mask = tensorlib.astensor([
            [
                [
                    mega_mods[s][m]['data']['mask'],
                ]
                for s in do_samples
            ] for m in histosys_mods
        ])
        self.histosys_default = tensorlib.zeros(self.histosys_mask.shape)

        self.histo_indices = tensorlib.astensor([
            self.parindices[pdfconfig.par_slice(m)] for m in histosys_mods
        ], dtype='int')

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
    def __init__(self,normfac_mods,pdf):
        pdfconfig = pdf.config
        mega_mods = pdf.mega_mods
        do_samples = pdf.do_samples


        self.parindices = list(range(len(pdfconfig.suggested_init())))
        tensorlib, _ = get_backend()
        self.normfactor_mask = tensorlib.astensor([
            [
                [
                    mega_mods[s][m]['data']['mask'],
                ]
                for s in do_samples
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
    def __init__(self,staterr_mods,pdf):
        pdfconfig = pdf.config
        do_channels = pdf.do_channels
        do_samples = pdf.do_samples
        channel_nbins = pdf.channel_nbins
        mega_mods = pdf.mega_mods

        start_index = 0
        channel_slices = []
        for c in do_channels:
            end_index = start_index + channel_nbins[c]
            channel_slices.append(slice(start_index,end_index))
            start_index = end_index

        binindices = list(range(sum(list(channel_nbins.values()))))
        channel_slice_map = {
            c:binindices[sl] for c,sl in zip(do_channels,channel_slices)
        }

        parindices = list(range(len(pdfconfig.suggested_init())))

        self._staterror_mask = default_backend.astensor([
            [
                [
                    mega_mods[s][m]['data']['mask'],
                ]
                for s in do_samples
            ] for m in staterr_mods
        ])
        self._staterror_default = default_backend.ones(default_backend.shape(self._staterror_mask))

        stat_parslices  = [pdfconfig.par_slice(m) for m in staterr_mods]
        stat_targetind  = [channel_slice_map[pdfconfig.modifier(m).channel] for m in staterr_mods]

        default_row =  [1.]*default_backend.shape(self._staterror_default)[-1]
        befores     = []
        afters      = []
        for sl,t in zip(stat_parslices,stat_targetind):
            before = default_backend.astensor(default_row[:t[0]])
            after  = default_backend.astensor(default_row[t[-1]+1:])
            befores.append(before)
            afters.append(after)
        if stat_parslices:
            tensorlib, _ = get_backend()
            factor_row_indices = default_backend.tolist(default_backend.stack([
                    default_backend.concatenate([before,default_backend.astensor(parindices[sl]),after])
                    for before,sl,after in zip(befores,stat_parslices,afters)
            ]))
            self.factor_row_indices = tensorlib.astensor(factor_row_indices,dtype='int')
            self.default_value = tensorlib.astensor([1.])
            self.sample_ones   = tensorlib.ones(len(do_samples))
            self.alpha_ones    = tensorlib.astensor([1])
            self.staterror_mask = default_backend.astensor(default_backend.tolist(self._staterror_mask))
            self.staterror_default = default_backend.astensor(default_backend.tolist(self._staterror_default))
        else:
            self.factor_row_indices = None


    def apply(self,pars):
        tensorlib, _ = get_backend()
        if self.factor_row_indices is None:
            return
        factor_row = tensorlib.gather(tensorlib.concatenate([pars,self.default_value]),self.factor_row_indices)
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
    def __init__(self,shapesys_mods,pdf):
        pdfconfig = pdf.config
        mega_mods = pdf.mega_mods
        do_channels = pdf.do_channels
        do_samples = pdf.do_samples
        channel_nbins = pdf.channel_nbins

        start_index = 0
        channel_slices = []
        for c in do_channels:
            end_index = start_index + channel_nbins[c]
            channel_slices.append(slice(start_index,end_index))
            start_index = end_index

        parindices = list(range(len(pdfconfig.suggested_init())))
        binindices = list(range(sum(list(channel_nbins.values()))))
        channel_slice_map = {
            c:binindices[sl] for c,sl in zip(do_channels,channel_slices)
        }

        self._shapesys_mask = default_backend.astensor([
            [
                [
                    mega_mods[s][m]['data']['mask'],
                ]
                for s in do_samples
            ] for m in shapesys_mods
        ])
        self._shapesys_default = default_backend.ones(self._shapesys_mask.shape)

        shapesys_parslices  = [pdfconfig.par_slice(m) for m in shapesys_mods]
        shapesys_targetind  = [
            channel_slice_map[pdfconfig.modifier(m).channel] for m in shapesys_mods
        ]
        default_row =  [1.]*self._shapesys_default.shape[-1]

        befores = []
        afters = []
        for sl,t in zip(shapesys_parslices,shapesys_targetind):
            before = default_backend.astensor(default_row[:t[0]])
            after  = default_backend.astensor(default_row[t[-1]+1:])
            befores.append(before)
            afters.append(after)

        if shapesys_parslices:
            tensorlib, _ = get_backend()
            factor_indices = default_backend.tolist(default_backend.stack([
                default_backend.concatenate([before,default_backend.astensor(parindices[sl]),after])
                for before,sl,after in zip(befores,shapesys_parslices,afters)
            ]))
            self.sample_ones = tensorlib.ones(len(do_samples))
            self.alpha_ones = tensorlib.astensor([1])
            self.factor_row_indices = tensorlib.astensor(factor_indices,dtype='int')
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

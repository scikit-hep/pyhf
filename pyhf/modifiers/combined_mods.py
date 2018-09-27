from .. import get_backend
from ..interpolate import _hfinterp_code1,_hfinterp_code0

class normsys_combinedmod(object):
    def __init__(self,normsys_mods,pdf):
        tensorlib, _ = get_backend()
        self.parindices = list(range(len(pdf.config.suggested_init())))
        self.normsys_histoset = tensorlib.astensor([
            [
                [
                    pdf.mega_mods[s][m]['data']['lo'],
                    [1.]*len(pdf.mega_samples[s]['nom']),
                    pdf.mega_mods[s][m]['data']['hi'],
                ]
                for s in pdf.do_samples
            ] for m in normsys_mods
        ])

        self.normsys_mask = tensorlib.astensor([
            [
                [
                    pdf.mega_mods[s][m]['data']['mask'],
                ]
                for s in pdf.do_samples
            ] for m in normsys_mods
        ])
        self.normsys_default = tensorlib.ones(self.normsys_mask.shape)
        

        self.normsys_indices = tensorlib.astensor([
            self.parindices[pdf.config.par_slice(m)] for m in normsys_mods
        ], dtype='int')
        

    def apply(self,pars):
        tensorlib, _ = get_backend()
        if not tensorlib.shape(self.normsys_indices)[0]:
            return
        normsys_alphaset = tensorlib.gather(pars,self.normsys_indices)
        results_norm   = _hfinterp_code1(self.normsys_histoset,normsys_alphaset)
        results_norm   = tensorlib.where(self.normsys_mask,results_norm,self.normsys_default)
        return results_norm


class histosys_combinedmod(object):
    def __init__(self,histosys_mods,pdf):
        tensorlib, _ = get_backend()
        self.parindices = list(range(len(pdf.config.suggested_init())))
        self.histosys_histoset = tensorlib.astensor([
            [
                [
                    pdf.mega_mods[s][m]['data']['lo_data'],
                    pdf.mega_samples[s]['nom'],
                    pdf.mega_mods[s][m]['data']['hi_data'],
                ]
                for s in pdf.do_samples
            ] for m in histosys_mods
        ])

        self.histosys_mask = tensorlib.astensor([
            [
                [
                    pdf.mega_mods[s][m]['data']['mask'],
                ]
                for s in pdf.do_samples
            ] for m in histosys_mods
        ])
        self.histosys_default = tensorlib.zeros(self.histosys_mask.shape)
        
        self.histo_indices = tensorlib.astensor([
            self.parindices[pdf.config.par_slice(m)] for m in histosys_mods
        ], dtype='int')

    def apply(self,pars):
        tensorlib, _ = get_backend()
        if not tensorlib.shape(self.histo_indices)[0]:
            return
        histosys_alphaset = tensorlib.gather(pars,self.histo_indices)
        results_histo   = _hfinterp_code0(self.histosys_histoset,histosys_alphaset)
        results_histo   = tensorlib.where(self.histosys_mask,results_histo,self.histosys_default)
        return results_histo


class normfac_combinedmod(object):
    def __init__(self,normfac_mods,pdf):
        self.parindices = list(range(len(pdf.config.suggested_init())))
        tensorlib, _ = get_backend()
        self.normfactor_mask = tensorlib.astensor([
            [
                [
                    pdf.mega_mods[s][m]['data']['mask'],
                ]
                for s in pdf.do_samples
            ] for m in normfac_mods
        ])
        self.normfactor_default = tensorlib.ones(self.normfactor_mask.shape)

        self.normfac_indices = tensorlib.astensor([self.parindices[pdf.config.par_slice(m)] for m in normfac_mods ], dtype='int')

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
        tensorlib, _ = get_backend()
        start_index = 0
        channel_slices = []
        for c in pdf.do_channels:
            end_index = start_index + pdf.channel_nbins[c]
            channel_slices.append(slice(start_index,end_index))
            start_index = end_index

        binindices = list(range(sum(list(pdf.channel_nbins.values()))))
        channel_slice_map = {
            c:binindices[sl] for c,sl in zip(pdf.do_channels,channel_slices)
        }

        self.parindices = list(range(len(pdf.config.suggested_init())))

        self.staterror_mask = tensorlib.astensor([
            [
                [
                    pdf.mega_mods[s][m]['data']['mask'],
                ]
                for s in pdf.do_samples
            ] for m in staterr_mods
        ])
        self.staterror_default = tensorlib.ones(self.staterror_mask.shape)

        self.stat_parslices  = [pdf.config.par_slice(m) for m in staterr_mods]
        self.stat_targetind  = [channel_slice_map[pdf.config.modifier(m).channel] for m in staterr_mods]

        self.sample_ones = tensorlib.ones(len(pdf.do_samples))
        self.alpha_ones = tensorlib.astensor([1])
        self.default_row =  [1.]*self.staterror_default.shape[-1]

        self.befores = []
        self.afters = []
        for sl,t in zip(self.stat_parslices,self.stat_targetind):
            before = tensorlib.astensor(self.default_row[:t[0]])
            after  = tensorlib.astensor(self.default_row[t[-1]+1:])
            self.befores.append(before)
            self.afters.append(after)

    def apply(self,pars):
        tensorlib, _ = get_backend()
        if not self.stat_parslices:
            return

        factor_row = tensorlib.stack([
            tensorlib.concatenate([before,pars[sl],after])
            for before,sl,after in zip(self.befores,self.stat_parslices,self.afters)
        ])

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
        tensorlib, _ = get_backend()
        start_index = 0
        channel_slices = []
        for c in pdf.do_channels:
            end_index = start_index + pdf.channel_nbins[c]
            channel_slices.append(slice(start_index,end_index))
            start_index = end_index

        binindices = list(range(sum(list(pdf.channel_nbins.values()))))
        channel_slice_map = {
            c:binindices[sl] for c,sl in zip(pdf.do_channels,channel_slices)
        }

        self.shapesys_mask = tensorlib.astensor([
            [
                [
                    pdf.mega_mods[s][m]['data']['mask'],
                ]
                for s in pdf.do_samples
            ] for m in shapesys_mods
        ])
        self.shapesys_default = tensorlib.ones(self.shapesys_mask.shape)

        self.shapesys_parslices  = [pdf.config.par_slice(m) for m in shapesys_mods]
        self.shapesys_targetind  = [
            channel_slice_map[pdf.config.modifier(m).channel] for m in shapesys_mods
        ]
        self.sample_ones = tensorlib.ones(len(pdf.do_samples))
        self.alpha_ones = tensorlib.astensor([1])
        self.default_row =  [1.]*self.shapesys_default.shape[-1]

        self.befores = []
        self.afters = []
        for sl,t in zip(self.shapesys_parslices,self.shapesys_targetind):
            before = tensorlib.astensor(self.default_row[:t[0]])
            after  = tensorlib.astensor(self.default_row[t[-1]+1:])
            self.befores.append(before)
            self.afters.append(after)

    def apply(self,pars):
        tensorlib, _ = get_backend()
        if not self.shapesys_parslices:
            return

        default = [1.]*self.shapesys_default.shape[-1]

        factor_row = tensorlib.stack([
            tensorlib.concatenate([before,pars[sl],after])
            for before,sl,after in zip(self.befores,self.shapesys_parslices,self.afters)
        ])

        results_shapesys = tensorlib.einsum('s,a,mb->msab',
                self.sample_ones,
                self.alpha_ones,
                factor_row)

        results_shapesys = tensorlib.where(self.shapesys_mask,results_shapesys,self.shapesys_default)
        return results_shapesys
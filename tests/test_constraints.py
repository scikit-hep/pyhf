import pytest
import pyhf

# @pytest.mark.skip_mxnet
def test_numpy_pdf_inputs(backend):
    spec = {
        'channels': [
            {
                'name': 'firstchannel',
                'samples': [
                    {
                        'name': 'mu',
                        'data': [10.,10.],
                        'modifiers': [
                            {'name': 'mu', 'type': 'normfactor', 'data': None}
                        ]
                    },
                    {
                        'name': 'bkg1',
                        'data': [50.0, 70.0],
                        'modifiers': [
                            {'name': 'stat_firstchannel', 'type': 'staterror', 'data': [12.,12.]}
                        ]
                    },
                    {
                        'name': 'bkg2',
                        'data': [30.0, 20.],
                        'modifiers': [
                            {'name': 'stat_firstchannel', 'type': 'staterror', 'data': [5.,5.]}
                        ]
                    },
                    {
                        'name': 'bkg3',
                        'data': [20.0, 15.0],
                        'modifiers': [
                            {'name': 'bkg_norm', 'type': 'shapesys','data': [10, 10]}
                        ]
                    }
                ]
            },
        ]
    }

    m = pyhf.Model(spec)
    def slow(self, auxdata, pars):
        tensorlib,_ = pyhf.get_backend()
        # iterate over all constraints order doesn't matter....
        start_index = 0
        summands = None
        for cname in self.config.auxdata_order:
            parset, parslice = self.config.param_set(cname), \
                self.config.par_slice(cname)
            paralphas = parset.alphas(pars[parslice])
            end_index = start_index + int(paralphas.shape[0])
            thisauxdata = auxdata[start_index:end_index]
            start_index = end_index
            if parset.pdf_type == 'normal':
                sigmas = parset.sigmas if hasattr(parset,'sigmas') else tensorlib.ones(paralphas.shape)
                sigmas = tensorlib.astensor(sigmas)
                constraint_term = tensorlib.normal_logpdf(thisauxdata, paralphas, sigmas)
            elif parset.pdf_type == 'poisson':
                constraint_term = tensorlib.poisson_logpdf(thisauxdata,paralphas)
            summands = constraint_term if summands is None else tensorlib.concatenate([summands,constraint_term])
        return tensorlib.sum(summands) if summands is not None else 0
    def fast(self, auxdata, pars):
        return self.constraint_logpdf(auxdata,pars)
    
    auxd = pyhf.tensorlib.astensor(m.config.auxdata)
    pars = pyhf.tensorlib.astensor(m.config.suggested_init())
    slow_result = pyhf.tensorlib.tolist(slow(m,auxd,pars))
    fast_result = pyhf.tensorlib.tolist(fast(m,auxd,pars))
    assert slow_result == fast_result
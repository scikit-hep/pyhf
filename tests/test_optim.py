from pyhf.optimize.opt_scipy import scipy_optimizer
from pyhf.tensor.pytorch_backend import pytorch_backend
from pyhf.tensor.numpy_backend import numpy_backend
from pyhf.optimize.opt_pytorch import pytorch_optimizer

def test_optim_numpy():
    import pyhf
    source = {
      "binning": [2,-0.5,1.5],
      "bindata": {
        "data":    [120.0, 180.0],
        "bkg":     [100.0, 150.0],
        "bkgsys_up":  [102, 190],
        "bkgsys_dn":  [98, 100],
        "sig":     [30.0, 95.0]
      }
    }
    spec = {
        'singlechannel': {
            'signal': {
                'data': source['bindata']['sig'],
                'mods': [
                    {
                        'name': 'mu',
                        'type': 'normfactor',
                        'data': None
                    }
                ]
            },
            'background': {
                'data': source['bindata']['bkg'],
                'mods': [
                    {
                        'name': 'bkg_norm',
                        'type': 'histosys',
                        'data': {
                            'lo_hist': source['bindata']['bkgsys_dn'],
                            'hi_hist': source['bindata']['bkgsys_up'],
                        }
                    }
                ]
            }
        }
    }
    pdf  = pyhf.hfpdf(spec)
    data = source['bindata']['data'] + pdf.config.auxdata

    init_pars  = pdf.config.suggested_init()
    par_bounds = pdf.config.suggested_bounds()

    oldlib = pyhf.tensorlib
    pyhf.tensorlib = pyhf.numpy_backend(poisson_from_normal = True)
    optim  = scipy_optimizer()

    v1 =  pdf.logpdf(init_pars,data)
    result = optim.unconstrained_bestfit(pyhf.loglambdav,data,pdf, init_pars, par_bounds)
    assert pyhf.tensorlib.tolist(result)
    pyhf.tensorlib = oldlib


def test_optim_pytorch():
    import pyhf
    source = {
      "binning": [2,-0.5,1.5],
      "bindata": {
        "data":    [120.0, 180.0],
        "bkg":     [100.0, 150.0],
        "bkgsys_up":  [102, 190],
        "bkgsys_dn":  [98, 100],
        "sig":     [30.0, 95.0]
      }
    }
    spec = {
        'singlechannel': {
            'signal': {
                'data': source['bindata']['sig'],
                'mods': [
                    {
                        'name': 'mu',
                        'type': 'normfactor',
                        'data': None
                    }
                ]
            },
            'background': {
                'data': source['bindata']['bkg'],
                'mods': [
                    {
                        'name': 'bkg_norm',
                        'type': 'histosys',
                        'data': {
                            'lo_hist': source['bindata']['bkgsys_dn'],
                            'hi_hist': source['bindata']['bkgsys_up'],
                        }
                    }
                ]
            }
        }
    }
    pdf  = pyhf.hfpdf(spec)
    data = source['bindata']['data'] + pdf.config.auxdata

    init_pars  = pdf.config.suggested_init()
    par_bounds = pdf.config.suggested_bounds()

    oldlib = pyhf.tensorlib

    pyhf.tensorlib = pyhf.pytorch_backend(poisson_from_normal = True)
    optim = pytorch_optimizer(tensorlib = pyhf.tensorlib)

    result = optim.unconstrained_bestfit(pyhf.loglambdav,data,pdf, init_pars, par_bounds)
    assert pyhf.tensorlib.tolist(result)


    pyhf.tensorlib = oldlib

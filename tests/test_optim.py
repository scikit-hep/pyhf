import pyhf

from pyhf.optimize.opt_scipy import scipy_optimizer
from pyhf.tensor.pytorch_backend import pytorch_backend
from pyhf.tensor.numpy_backend import numpy_backend
from pyhf.optimize.opt_pytorch import pytorch_optimizer
from pyhf.optimize.opt_tflow import tflow_optimizer

import tensorflow as tf


def test_optim_numpy():
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
        'channels': [
            {
                'name': 'singlechannel',
                'samples': [
                    {
                        'name': 'signal',
                        'data': source['bindata']['sig'],
                        'modifiers': [
                            {'name': 'mu', 'type': 'normfactor', 'data': None}
                        ]
                    },
                    {
                        'name': 'background',
                        'data': source['bindata']['bkg'],
                        'modifiers': [
                            {'name': 'bkg_norm', 'type': 'histosys', 'data': {'lo_data': source['bindata']['bkgsys_dn'], 'hi_data': source['bindata']['bkgsys_up']}}
                        ]
                    }
                ]
            }
        ]
    }
    pdf = pyhf.hfpdf(spec)
    data = source['bindata']['data'] + pdf.config.auxdata

    init_pars = pdf.config.suggested_init()
    par_bounds = pdf.config.suggested_bounds()

    oldlib = pyhf.tensorlib
    pyhf.set_backend(pyhf.numpy_backend(poisson_from_normal=True))
    optim = pyhf.optimizer

    v1 = pdf.logpdf(init_pars, data)
    result = optim.unconstrained_bestfit(pyhf.loglambdav, data, pdf, init_pars, par_bounds)
    assert pyhf.tensorlib.tolist(result)

    result = optim.constrained_bestfit(pyhf.loglambdav, 1.0, data, pdf, init_pars, par_bounds)
    assert pyhf.tensorlib.tolist(result)

    pyhf.set_backend(oldlib)


def test_optim_pytorch():
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
        'channels': [
            {
                'name': 'singlechannel',
                'samples': [
                    {
                        'name': 'signal',
                        'data': source['bindata']['sig'],
                        'modifiers': [
                            {'name': 'mu', 'type': 'normfactor', 'data': None}
                        ]
                    },
                    {
                        'name': 'background',
                        'data': source['bindata']['bkg'],
                        'modifiers': [
                            {'name': 'bkg_norm', 'type': 'histosys', 'data': {'lo_data': source['bindata']['bkgsys_dn'], 'hi_data': source['bindata']['bkgsys_up']}}
                        ]
                    }
                ]
            }
        ]
    }
    pdf = pyhf.hfpdf(spec)
    data = source['bindata']['data'] + pdf.config.auxdata

    init_pars = pdf.config.suggested_init()
    par_bounds = pdf.config.suggested_bounds()

    oldlib = pyhf.tensorlib

    pyhf.set_backend(pyhf.pytorch_backend(poisson_from_normal=True))
    optim = pyhf.optimizer

    result = optim.unconstrained_bestfit(pyhf.loglambdav, data, pdf, init_pars, par_bounds)
    assert pyhf.tensorlib.tolist(result)

    result = optim.constrained_bestfit(pyhf.loglambdav, 1.0, data, pdf, init_pars, par_bounds)
    assert pyhf.tensorlib.tolist(result)

    pyhf.set_backend(oldlib)


def test_optim_tflow():
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
        'channels': [
            {
                'name': 'singlechannel',
                'samples': [
                    {
                        'name': 'signal',
                        'data': source['bindata']['sig'],
                        'modifiers': [
                            {'name': 'mu', 'type': 'normfactor', 'data': None}
                        ]
                    },
                    {
                        'name': 'background',
                        'data': source['bindata']['bkg'],
                        'modifiers': [
                            {'name': 'bkg_norm', 'type': 'histosys', 'data': {'lo_data': source['bindata']['bkgsys_dn'], 'hi_data': source['bindata']['bkgsys_up']}}
                        ]
                    }
                ]
            }
        ]
    }
    pdf = pyhf.hfpdf(spec)
    data = source['bindata']['data'] + pdf.config.auxdata

    init_pars = pdf.config.suggested_init()
    par_bounds = pdf.config.suggested_bounds()

    oldlib = pyhf.tensorlib

    pyhf.set_backend(pyhf.tensorflow_backend())
    pyhf.tensorlib.session = tf.Session()
    optim = pyhf.optimizer

    result = optim.unconstrained_bestfit(pyhf.loglambdav, data, pdf, init_pars, par_bounds)
    assert pyhf.tensorlib.tolist(result)

    result = optim.constrained_bestfit(pyhf.loglambdav, 1.0, data, pdf, init_pars, par_bounds)
    assert pyhf.tensorlib.tolist(result)

    pyhf.set_backend(oldlib)

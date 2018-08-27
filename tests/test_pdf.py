import pyhf
import pytest
import pyhf.simplemodels
import pyhf.exceptions
import numpy as np
import json
import tensorflow as tf

def test_numpy_pdf_inputs():
    source = {
      "binning": [2,-0.5,1.5],
      "bindata": {
        "data":    [55.0],
        "bkg":     [50.0],
        "bkgerr":  [7.0],
        "sig":     [10.0]
      }
    }
    pdf  = pyhf.simplemodels.hepdata_like(source['bindata']['sig'], source['bindata']['bkg'], source['bindata']['bkgerr'])

    pars = pdf.config.suggested_init()
    data = source['bindata']['data'] + pdf.config.auxdata


    np_data       = np.array(data)
    np_parameters = np.array(pars)

    assert len(data) == np_data.shape[0]
    assert len(pars) == np_parameters.shape[0]
    assert pdf.pdf(pars,data) == pdf.pdf(np_parameters,np_data)
    assert pdf.logpdf(pars,data) == pdf.logpdf(np_parameters,np_data)
    assert np.array(pdf.logpdf(np_parameters,np_data)).shape == (1,)


def test_core_pdf_broadcasting():
    data    = [10,11,12,13,14,15]
    lambdas = [15,14,13,12,11,10]
    naive_python = [pyhf.tensorlib.poisson(d, lam) for d,lam in zip(data, lambdas)]

    broadcasted  = pyhf.tensorlib.poisson(data, lambdas)

    assert np.array(data).shape == np.array(lambdas).shape
    assert broadcasted.shape    == np.array(data).shape
    assert np.all(naive_python  == broadcasted)


    data    = [10,11,12,13,14,15]
    mus     = [15,14,13,12,11,10]
    sigmas  = [1,2,3,4,5,6]
    naive_python = [pyhf.tensorlib.normal(d, mu,sig) for d,mu,sig in zip(data, mus, sigmas)]

    broadcasted  = pyhf.tensorlib.normal(data, mus, sigmas)

    assert np.array(data).shape == np.array(mus).shape
    assert np.array(data).shape == np.array(sigmas).shape
    assert broadcasted.shape    == np.array(data).shape
    assert np.all(naive_python  == broadcasted)

def test_pdf_integration_staterror():
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
                        ]
                    }
                ]
            },
        ]
    }
    pdf = pyhf.Model(spec)
    par = pdf.config.par_slice('stat_firstchannel')
    mod = pdf.config.modifier('stat_firstchannel')
    assert mod.uncertainties == [[12.,12.],[5.,5.]]
    assert mod.nominal_counts == [[50.,70.],[30.,20.]]

    computed = pyhf.tensorlib.tolist(mod.pdf([1.0,1.0],[1.0,1.0]))
    expected = pyhf.tensorlib.tolist(pyhf.tensorlib.normal([1.0,1.0], mu = [1.0,1.0], sigma = [13./80.,13./90.]))
    for c,e in zip(computed,expected):
        assert c==e

def test_pdf_integration_histosys():
    source = json.load(open('validation/data/2bin_histosys_example2.json'))
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
    pdf  = pyhf.Model(spec)


    pars = [None,None]

    pars[pdf.config.par_slice('mu')], pars[pdf.config.par_slice('bkg_norm')] = [[0.0], [1.0]]
    assert pdf.expected_data(pars, include_auxdata = False).tolist() == [102,190]


    pars[pdf.config.par_slice('mu')], pars[pdf.config.par_slice('bkg_norm')] = [[0.0], [2.0]]
    assert pdf.expected_data(pars, include_auxdata = False).tolist()  == [104,230]

    pars[pdf.config.par_slice('mu')], pars[pdf.config.par_slice('bkg_norm')] = [[0.0], [-1.0]]
    assert pdf.expected_data(pars, include_auxdata = False).tolist() == [ 98,100]

    pars[pdf.config.par_slice('mu')], pars[pdf.config.par_slice('bkg_norm')] = [[0.0], [-2.0]]
    assert pdf.expected_data(pars, include_auxdata = False).tolist() == [ 96, 50]

    pars[pdf.config.par_slice('mu')], pars[pdf.config.par_slice('bkg_norm')] = [[1.0], [1.0]]
    assert pdf.expected_data(pars, include_auxdata = False).tolist()  == [102+30,190+95]

    pars[pdf.config.par_slice('mu')], pars[pdf.config.par_slice('bkg_norm')] = [[1.0], [-1.0]]
    assert pdf.expected_data(pars, include_auxdata = False).tolist() == [ 98+30,100+95]


@pytest.mark.parametrize('backend',
                         [
                             pyhf.tensor.numpy_backend(poisson_from_normal=True),
                             pyhf.tensor.tensorflow_backend(session=tf.Session()),
                             pyhf.tensor.pytorch_backend(poisson_from_normal=True),
                             # pyhf.tensor.mxnet_backend(),
                         ],
                         ids=[
                             'numpy',
                             'tensorflow',
                             'pytorch',
                         ])
def test_pdf_integration_normsys(backend):
    pyhf.set_backend(backend)
    if isinstance(pyhf.tensorlib, pyhf.tensor.tensorflow_backend):
        tf.reset_default_graph()
        pyhf.tensorlib.session = tf.Session()
    source = json.load(open('validation/data/2bin_histosys_example2.json'))
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
                            {'name': 'bkg_norm', 'type': 'normsys','data': {'lo': 0.9, 'hi': 1.1}}
                        ]
                    }
                ]
            }
        ]
    }
    pdf  = pyhf.Model(spec)

    pars = [None,None]
    pars[pdf.config.par_slice('mu')], pars[pdf.config.par_slice('bkg_norm')] = [[0.0], [0.0]]
    assert np.allclose(pyhf.tensorlib.tolist(pdf.expected_data(pars, include_auxdata = False)),[100,150])

    pars[pdf.config.par_slice('mu')], pars[pdf.config.par_slice('bkg_norm')] = [[0.0], [1.0]]
    assert np.allclose(pyhf.tensorlib.tolist(pdf.expected_data(pars, include_auxdata = False)),[100*1.1,150*1.1])

    pars[pdf.config.par_slice('mu')], pars[pdf.config.par_slice('bkg_norm')] = [[0.0], [-1.0]]
    assert np.allclose(pyhf.tensorlib.tolist(pdf.expected_data(pars, include_auxdata = False)),[100*0.9,150*0.9])

def test_pdf_integration_shapesys():
    source = json.load(open('validation/data/2bin_histosys_example2.json'))
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
                            {'name': 'bkg_norm', 'type': 'shapesys','data': [10, 10]}
                        ]
                    }
                ]
            }
        ]
    }
    pdf  = pyhf.Model(spec)



    pars = [None,None]


    pars[pdf.config.par_slice('mu')], pars[pdf.config.par_slice('bkg_norm')] = [[0.0], [1.0,1.0]]
    assert pdf.expected_data(pars, include_auxdata = False).tolist()   == [100,150]

    pars[pdf.config.par_slice('mu')], pars[pdf.config.par_slice('bkg_norm')] = [[0.0], [1.1,1.0]]
    assert pdf.expected_data(pars, include_auxdata = False).tolist()   == [100*1.1,150]

    pars[pdf.config.par_slice('mu')], pars[pdf.config.par_slice('bkg_norm')] = [[0.0], [1.0,1.1]]
    assert pdf.expected_data(pars, include_auxdata = False).tolist()   == [100,150*1.1]

    pars[pdf.config.par_slice('mu')], pars[pdf.config.par_slice('bkg_norm')] = [[0.0], [1.1, 0.9]]
    assert pdf.expected_data(pars, include_auxdata = False).tolist()   == [100*1.1,150*0.9]

    pars[pdf.config.par_slice('mu')], pars[pdf.config.par_slice('bkg_norm')] = [[0.0], [0.9,1.1]]
    assert pdf.expected_data(pars, include_auxdata = False).tolist()   == [100*0.9,150*1.1]

def test_invalid_modifier():
    spec = {
        'channels': [
            {
                'name': 'channel',
                'samples': [
                    {
                        'name': 'ttbar',
                        'data': [1],
                        'modifiers': [
                            {'name': 'a_name', 'type': 'this_should_not_exist', 'data': [1]}
                        ]
                    },
                ]
            }
        ]
    }
    with pytest.raises(pyhf.exceptions.InvalidModifier):
        pyhf.pdf._ModelConfig.from_spec(spec)

def test_invalid_modifier_name_resuse():
    spec = {
        'channels': [
            {
                'name': 'singlechannel',
                'samples': [
                    {
                        'name': 'signal',
                        'data': [5.],
                        'modifiers': [
                            {'name': 'reused_name', 'type': 'normfactor', 'data': None}
                        ]
                    },
                    {
                        'name': 'background',
                        'data': [50.],
                        'modifiers': [
                            {'name': 'reused_name', 'type': 'normsys','data': {'lo': 0.9, 'hi': 1.1}}
                        ]
                    }
                ]
            }
        ]
    }
    with pytest.raises(pyhf.exceptions.InvalidNameReuse):
        pdf  = pyhf.Model(spec, poiname = 'reused_name')

    pdf  = pyhf.Model(spec, poiname = 'reused_name', qualify_names = True)
    
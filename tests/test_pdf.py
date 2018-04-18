import pyhf
import pyhf.simplemodels
import numpy as np
import json
import jsonschema

def test_interpcode_0():
    f = lambda x: pyhf._hfinterp_code0(at_minus_one = 0.5, at_zero =1, at_plus_one = 2.0, alphas = x)
    assert 1+f(-2) == 0.0
    assert 1+f(-1) == 0.5
    assert 1+f(0) == 1.0
    assert 1+f(1) == 2.0
    assert 1+f(2) == 3.0 #extrapolation

    #broadcasting
    assert [1 + x for x in f([-2,-1,0,1,2]).reshape(-1)] == [0,0.5,1.0,2.0,3.0]

def test_interpcode_1():
    f = lambda x: pyhf._hfinterp_code1(at_minus_one = 0.9, at_zero =1, at_plus_one = 1.1, alphas = x)
    assert f(-2) == 0.9**2
    assert f(-1) == 0.9
    assert f(0) == 1.0
    assert f(1) == 1.1
    assert f(2) == 1.1**2

    #broadcasting
    assert np.all(f([-2,-1,0,1,2]).reshape(-1) == [0.9**2, 0.9, 1.0, 1.1, 1.1**2])

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

def test_add_unknown_modifier():
    spec = {
        'channels': [
            {
                'name': 'channe',
                'samples': [
                    {
                        'modifiers': [
                            {'name': 'a_name', 'type': 'this_should_not_exist', 'data': None}
                        ]
                    },
                ]
            }
        ]
    }
    pyhf.hfpdf(spec)


def test_pdf_integration_histosys():
    schema = json.load(open('validation/spec.json'))
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
    jsonschema.validate(spec, schema)
    pdf  = pyhf.hfpdf(spec)


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


def test_pdf_integration_normsys():
    schema = json.load(open('validation/spec.json'))
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
    jsonschema.validate(spec, schema)
    pdf  = pyhf.hfpdf(spec)

    pars = [None,None]
    pars[pdf.config.par_slice('mu')], pars[pdf.config.par_slice('bkg_norm')] = [[0.0], [0.0]]
    assert pdf.expected_data(pars, include_auxdata = False).tolist()   == [100,150]

    pars[pdf.config.par_slice('mu')], pars[pdf.config.par_slice('bkg_norm')] = [[0.0], [1.0]]
    assert pdf.expected_data(pars, include_auxdata = False).tolist()   == [100*1.1,150*1.1]

    pars[pdf.config.par_slice('mu')], pars[pdf.config.par_slice('bkg_norm')] = [[0.0], [-1.0]]
    assert pdf.expected_data(pars, include_auxdata = False).tolist()   == [100*0.9,150*0.9]

def test_pdf_integration_shapesys():
    schema = json.load(open('validation/spec.json'))
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
    jsonschema.validate(spec, schema)
    pdf  = pyhf.hfpdf(spec)



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

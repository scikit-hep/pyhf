import pyhf
import pyhf.simplemodels
import numpy as np
import json

VALIDATION_TOLERANCE = 1e-5

def test_validation_1bin_shapesys():
    expected_result = {
        'obs': 0.450337178157,
        'exp': [
            0.06154653039922158,
            0.1472337570386738,
            0.3227412178815565,
            0.5995781547454528,
            0.8636787737204704
        ]
    }


    source = json.load(open('validation/data/1bin_example1.json'))
    pdf  = pyhf.simplemodels.hepdata_like(source['bindata']['sig'], source['bindata']['bkg'], source['bindata']['bkgerr'])

    data = source['bindata']['data'] + pdf.auxdata
    muTest = 1.0

    assert len(pdf.config.suggested_init()) == 2
    assert len(pdf.config.suggested_bounds()) == 2
    init_pars  = pdf.config.suggested_init()
    par_bounds = pdf.config.suggested_bounds()

    clsobs, cls_exp = pyhf.runOnePoint(muTest, data,pdf,init_pars,par_bounds)[-2:]
    cls_obs = 1./clsobs
    cls_exp = [1./x for x in cls_exp]
    assert (cls_obs - expected_result['obs'])/expected_result['obs'] < VALIDATION_TOLERANCE
    for result,expected_result in zip(cls_exp, expected_result['exp']):
        assert (result-expected_result)/expected_result < VALIDATION_TOLERANCE


def test_validation_1bin_normsys():
    expected_result = {
        'obs': 0.0007930094233140433,
        'exp': [
            1.2529050370718884e-09,
            8.932001833559302e-08,
            5.3294967286010575e-06,
            0.00022773982308763686,
            0.0054897420571466075
        ]
    }
    source = {
      "binning": [2,-0.5,1.5],
      "bindata": {
        "data":    [120.0, 180.0],
        "bkg":     [100.0, 150.0],
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
                        'type': 'normsys',
                        'data': {'lo': 0.90, 'hi': 1.10}
                    }
                ]
            }
        }
    }
    pdf  = pyhf.hfpdf(spec)

    data = source['bindata']['data'] + pdf.auxdata

    muTest = 1.0

    assert len(pdf.config.suggested_init()) == 2
    assert len(pdf.config.suggested_bounds()) == 2
    init_pars  = pdf.config.suggested_init()
    par_bounds = pdf.config.suggested_bounds()


    clsobs, cls_exp = pyhf.runOnePoint(muTest, data,pdf,init_pars,par_bounds)[-2:]
    cls_obs = 1./clsobs
    cls_exp = [1./x for x in cls_exp]
    assert (cls_obs - expected_result['obs'])/expected_result['obs'] < VALIDATION_TOLERANCE
    for result,expected_result in zip(cls_exp, expected_result['exp']):
        assert (result-expected_result)/expected_result < VALIDATION_TOLERANCE


def test_validation_2bin_histosys():
    expected_result = {
        'obs': 0.09436700514736625,
        'exp': [
            8.131143652258812e-06,
            0.0001396307700293439,
            0.0020437905684851376,
            0.022094931468776054,
            0.14246926685789288,
        ]
    }
    source = json.load(open('validation/data/2bin_histosys_example2.json'))
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

    data = source['bindata']['data'] + pdf.auxdata

    muTest = 1.0

    assert len(pdf.config.suggested_init()) == 2
    assert len(pdf.config.suggested_bounds()) == 2
    init_pars  = pdf.config.suggested_init()
    par_bounds = pdf.config.suggested_bounds()


    clsobs, cls_exp = pyhf.runOnePoint(muTest, data,pdf,init_pars,par_bounds)[-2:]
    cls_obs = 1./clsobs
    cls_exp = [1./x for x in cls_exp]
    assert (cls_obs - expected_result['obs'])/expected_result['obs'] < VALIDATION_TOLERANCE
    for result,expected_result in zip(cls_exp, expected_result['exp']):
        assert (result-expected_result)/expected_result < VALIDATION_TOLERANCE



def test_validation_2bin_2channel():
    expected_result = {
        'obs': 0.05691881515460979,
        'exp': [
            0.0004448774256747925,
            0.0034839534635069816,
            0.023684793938725246,
            0.12294326553585197,
            0.4058143629613449
        ]
    }
    source = json.load(open('validation/data/2bin_2channel_example1.json'))
    spec =  {
        'signal': {
            'signal': {
                'data': source['channels']['signal']['bindata']['sig'],
                'mods': [
                    {
                        'name': 'mu',
                        'type': 'normfactor',
                        'data': None
                    }
                ]
            },
            'background': {
                'data': source['channels']['signal']['bindata']['bkg'],
                'mods': [
                    {
                        'name': 'uncorr_bkguncrt_signal',
                        'type': 'shapesys',
                        'data': source['channels']['signal']['bindata']['bkgerr']
                    }
                ]
            }
        },
        'control': {
            'background': {
                'data': source['channels']['control']['bindata']['bkg'],
                'mods': [
                    {
                        'name': 'uncorr_bkguncrt_control',
                        'type': 'shapesys',
                        'data': source['channels']['control']['bindata']['bkgerr']
                    }
                ]
            }
        }
    }
    pdf  = pyhf.hfpdf(spec)
    data = []
    for c in pdf.channel_order:
        data += source['channels'][c]['bindata']['data']
    data = data + pdf.auxdata

    muTest = 1.0

    assert len(pdf.config.suggested_init())   == 5 # 1 mu + 2 gammas for 2 channels each
    assert len(pdf.config.suggested_bounds()) == 5
    init_pars  = pdf.config.suggested_init()
    par_bounds = pdf.config.suggested_bounds()

    clsobs, cls_exp = pyhf.runOnePoint(muTest, data,pdf,init_pars,par_bounds)[-2:]
    cls_obs = 1./clsobs
    cls_exp = [1./x for x in cls_exp]
    assert (cls_obs - expected_result['obs'])/expected_result['obs'] < VALIDATION_TOLERANCE
    for result,expected_result in zip(cls_exp, expected_result['exp']):
        assert (result-expected_result)/expected_result < VALIDATION_TOLERANCE



def test_validation_2bin_2channel_couplednorm():
    expected_result = {
        'obs': 0.5999662863185762,
        'exp': [0.06596134134354742,
          0.15477912571478988,
          0.33323967895587736,
          0.6096429330789306,
          0.8688213053042003
        ]
    }
    source = json.load(open('validation/data/2bin_2channel_couplednorm.json'))
    spec = {
        'signal': {
            'signal': {
                'data': source['channels']['signal']['bindata']['sig'],
                'mods': [
                    {
                        'name': 'mu',
                        'type': 'normfactor',
                        'data': None
                    }
                ]
            },
            'bkg1': {
                'data': source['channels']['signal']['bindata']['bkg1'],
                'mods': [
                    {
                        'name': 'coupled_normsys',
                        'type': 'normsys',
                        'data':  {'lo': 0.9, 'hi': 1.1}
                    }
                ]
            },
            'bkg2': {
                'data': source['channels']['signal']['bindata']['bkg2'],
                'mods': [
                    {
                        'name': 'coupled_normsys',
                        'type': 'normsys',
                        'data':  {'lo': 0.5, 'hi': 1.5}
                    }
                ]
            }
        },
        'control': {
            'background': {
                'data': source['channels']['control']['bindata']['bkg1'],
                'mods': [
                    {
                        'name': 'coupled_normsys',
                        'type': 'normsys',
                        'data': {'lo': 0.9, 'hi': 1.1}
                    }
                ]
            }
        }
    }
    pdf  = pyhf.hfpdf(spec)
    data = []
    for c in pdf.channel_order:
        data += source['channels'][c]['bindata']['data']
    data = data + pdf.auxdata


    muTest = 1.0
    assert len(pdf.config.suggested_init())   == 2 # 1 mu + 1 alpha
    assert len(pdf.config.suggested_bounds()) == 2 # 1 mu + 1 alpha
    init_pars  = pdf.config.suggested_init()
    par_bounds = pdf.config.suggested_bounds()

    clsobs, cls_exp = pyhf.runOnePoint(muTest, data,pdf,init_pars,par_bounds)[-2:]
    cls_obs = 1./clsobs
    cls_exp = [1./x for x in cls_exp]
    assert (cls_obs - expected_result['obs'])/expected_result['obs'] < VALIDATION_TOLERANCE
    for result,expected_result in zip(cls_exp, expected_result['exp']):
        assert (result-expected_result)/expected_result < VALIDATION_TOLERANCE

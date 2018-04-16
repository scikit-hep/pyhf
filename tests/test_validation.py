import pyhf
import pyhf.simplemodels
import numpy as np
import json

VALIDATION_TOLERANCE = 1e-5

def test_validation_1bin_shapesys():
    expected_result = {
        'obs': 0.4541865416107029,
        'exp': [
            0.06371799398864626,
            0.15096503398048894,
            0.3279606950533305,
            0.6046087303039118,
            0.8662627605298466
        ]
    }


    source = json.load(open('validation/data/1bin_example1.json'))
    pdf  = pyhf.simplemodels.hepdata_like(source['bindata']['sig'], source['bindata']['bkg'], source['bindata']['bkgerr'])

    data = source['bindata']['data'] + pdf.config.auxdata
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
                            {'name': 'bkg_norm', 'type': 'normsys', 'data': {'lo': 0.90, 'hi': 1.10}}
                        ]
                    }
                ]
            }
        ]
    }
    pdf  = pyhf.hfpdf(spec)

    data = source['bindata']['data'] + pdf.config.auxdata

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
        'obs': 0.10014623469489856,
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
    pdf  = pyhf.hfpdf(spec)

    data = source['bindata']['data'] + pdf.config.auxdata

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
        'channels': [
            {
                'name': 'signal',
                'samples': [
                    {
                        'name': 'signal',
                        'data': source['channels']['signal']['bindata']['sig'],
                        'modifiers': [
                            {'name': 'mu', 'type': 'normfactor', 'data': None}
                        ]
                    },
                    {
                        'name': 'background',
                        'data': source['channels']['signal']['bindata']['bkg'],
                        'modifiers': [
                            {'name': 'uncorr_bkguncrt_signal', 'type': 'shapesys', 'data': source['channels']['signal']['bindata']['bkgerr']}
                        ]
                    }
                ]
            },
            {
                'name': 'control',
                'samples': [
                    {
                        'name': 'background',
                        'data': source['channels']['control']['bindata']['bkg'],
                        'modifiers': [
                            {'name': 'uncorr_bkguncrt_control', 'type': 'shapesys', 'data': source['channels']['control']['bindata']['bkgerr']}
                        ]
                    }
                ]
            }
        ]
    }
    pdf  = pyhf.hfpdf(spec)
    data = []
    for c in pdf.spec['channels']:
        data += source['channels'][c['name']]['bindata']['data']
    data = data + pdf.config.auxdata

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
    spec =  {
        'channels': [
            {
                'name': 'signal',
                'samples': [
                    {
                        'name': 'signal',
                        'data': source['channels']['signal']['bindata']['sig'],
                        'modifiers': [
                            {'name': 'mu', 'type': 'normfactor', 'data': None}
                        ]
                    },
                    {
                        'name': 'bkg1',
                        'data': source['channels']['signal']['bindata']['bkg1'],
                        'modifiers': [
                            {'name': 'coupled_normsys', 'type': 'normsys', 'data':  {'lo': 0.9, 'hi': 1.1}}
                        ]
                    },
                    {
                        'name': 'bkg2',
                        'data': source['channels']['signal']['bindata']['bkg2'],
                        'modifiers': [
                            {'name': 'coupled_normsys', 'type': 'normsys', 'data':  {'lo': 0.5, 'hi': 1.5}}
                        ]
                    }
                ]
            },
            {
                'name': 'control',
                'samples': [
                    {
                        'name': 'background',
                        'data': source['channels']['control']['bindata']['bkg1'],
                        'modifiers': [
                            {'name': 'coupled_normsys', 'type': 'normsys', 'data': {'lo': 0.9, 'hi': 1.1}}
                        ]
                    }
                ]
            }
        ]
    }
    pdf  = pyhf.hfpdf(spec)
    data = []
    for c in pdf.spec['channels']:
        data += source['channels'][c['name']]['bindata']['data']
    data = data + pdf.config.auxdata


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



def test_validation_2bin_2channel_coupledhistosys():
    expected_result = {
    'obs': 0.0796739833305826,
     'exp': [
        1.765372502072074e-05,
        0.00026265618793683054,
        0.003340033567379219,
        0.03152233566143051,
        0.17907736639946248
    ]
    }
    source = json.load(open('validation/data/2bin_2channel_coupledhisto.json'))
    spec   =  {
        'channels': [
            {
                'name': 'signal',
                'samples': [
                    {
                        'name': 'signal',
                        'data': source['channels']['signal']['bindata']['sig'],
                        'modifiers': [
                            {'name': 'mu', 'type': 'normfactor', 'data': None}
                        ]
                    },
                    {
                        'name': 'bkg1',
                        'data': source['channels']['signal']['bindata']['bkg1'],
                        'modifiers': [
                            {'name': 'coupled_histosys','type': 'histosys', 'data': {'lo_data': source['channels']['signal']['bindata']['bkg1_dn'], 'hi_data': source['channels']['signal']['bindata']['bkg1_up']}}
                        ]
                    },
                    {
                        'name': 'bkg2',
                        'data': source['channels']['signal']['bindata']['bkg2'],
                        'modifiers': [
                            {'name': 'coupled_histosys', 'type': 'histosys', 'data': {'lo_data': source['channels']['signal']['bindata']['bkg2_dn'], 'hi_data': source['channels']['signal']['bindata']['bkg2_up']}}
                        ]
                    }
                ]
            },
            {
                'name': 'control',
                'samples': [
                    {
                        'name': 'background',
                        'data': source['channels']['control']['bindata']['bkg1'],
                        'modifiers': [
                            {'name': 'coupled_histosys', 'type': 'histosys', 'data': {'lo_data': source['channels']['control']['bindata']['bkg1_dn'], 'hi_data': source['channels']['control']['bindata']['bkg1_up']}}
                        ]
                    }
                ]
            }
        ]
    }
    pdf  = pyhf.hfpdf(spec)
    data = []
    for c in pdf.spec['channels']:
        data += source['channels'][c['name']]['bindata']['data']
    data = data + pdf.config.auxdata

    init_pars = pdf.config.suggested_init()
    par_bounds = pdf.config.suggested_bounds()

    assert len(pdf.config.auxdata) == 1
    assert len(init_pars)  == 2 #1 mu 1 shared histosys
    assert len(par_bounds) == 2

    muTest = 1.0
    clsobs, cls_exp = pyhf.runOnePoint(muTest, data,pdf,init_pars,par_bounds)[-2:]
    cls_obs = 1./clsobs
    cls_exp = [1./x for x in cls_exp]
    assert (cls_obs - expected_result['obs'])/expected_result['obs'] < VALIDATION_TOLERANCE
    for result,expected_result in zip(cls_exp, expected_result['exp']):
        assert (result-expected_result)/expected_result < VALIDATION_TOLERANCE


def test_validation_2bin_2channel_coupledshapefactor():
    expected_result = {
    'obs': 0.5421679124909312,
     'exp': [
        0.013753299929451691,
        0.048887400056355966,
        0.15555296253957684,
        0.4007561343326305,
        0.7357169630955912
        ]
    }
    source = json.load(open('validation/data/2bin_2channel_coupledshapefactor.json'))
    spec =  {
        'channels': [
            {
                'name': 'signal',
                'samples': [
                    {
                        'name': 'signal',
                        'data': source['channels']['signal']['bindata']['sig'],
                        'modifiers': [
                            {'name': 'mu', 'type': 'normfactor', 'data': None}
                        ]
                    },
                    {
                        'name': 'bkg1',
                        'data': source['channels']['signal']['bindata']['bkg1'],
                        'modifiers': [
                            {'name': 'coupled_shapefactor', 'type': 'shapefactor', 'data': None}
                        ]
                    }
                ]
            },
            {
                'name': 'control',
                'samples': [
                    {
                        'name': 'background',
                        'data': source['channels']['control']['bindata']['bkg1'],
                        'modifiers': [
                            {'name': 'coupled_shapefactor', 'type': 'shapefactor', 'data': None}
                        ]
                    }
                ]
            }
        ]
    }
    pdf  = pyhf.hfpdf(spec)
    data = []
    for c in pdf.spec['channels']:
        data += source['channels'][c['name']]['bindata']['data']
    data = data + pdf.config.auxdata

    init_pars = pdf.config.suggested_init()
    par_bounds = pdf.config.suggested_bounds()

    assert len(pdf.config.auxdata) == 0
    assert len(init_pars)  == 3 #1 mu 2 shared shapefactors
    assert len(par_bounds) == 3

    muTest = 1.0
    clsobs, cls_exp = pyhf.runOnePoint(muTest, data,pdf,init_pars,par_bounds)[-2:]
    cls_obs = 1./clsobs
    cls_exp = [1./x for x in cls_exp]
    assert (cls_obs - expected_result['obs'])/expected_result['obs'] < VALIDATION_TOLERANCE
    for result,expected_result in zip(cls_exp, expected_result['exp']):
        assert (result-expected_result)/expected_result < VALIDATION_TOLERANCE

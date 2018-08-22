import pyhf
import json
import pytest

@pytest.fixture(scope='module')
def source_1bin_example1():
    return json.load(open('validation/data/1bin_example1.json'))


@pytest.fixture(scope='module')
def spec_1bin_shapesys(source=source_1bin_example1()):
    spec = {
        'channels': [
            {
                'name': 'singlechannel',
                'samples': [
                    {
                        'name': 'signal',
                        'data': source['bindata']['sig'],
                        'modifiers': [
                            {
                                'name': 'mu',
                                'type': 'normfactor',
                                'data': None
                            }
                        ]
                    },
                    {
                        'name': 'background',
                        'data': source['bindata']['bkg'],
                        'modifiers': [
                            {
                                'name': 'uncorr_bkguncrt',
                                'type': 'shapesys',
                                'data': source['bindata']['bkgerr']
                            }
                        ]
                    }
                ]
            }
        ]
    }
    return spec


@pytest.fixture(scope='module')
def expected_result_1bin_shapesys(mu=1.):
    if mu == 1:
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
    return expected_result


@pytest.fixture(scope='module')
def setup_1bin_shapesys(source=source_1bin_example1(),
                        spec=spec_1bin_shapesys(source_1bin_example1()),
                        mu=1,
                        expected_result=expected_result_1bin_shapesys(1.),
                        config={'init_pars': 2, 'par_bounds': 2}):
    return {
        'source': source,
        'spec': spec,
        'mu': mu,
        'expected': {
            'result': expected_result,
            'config': config
        }
    }


@pytest.fixture(scope='module')
def source_1bin_normsys():
    source = {
        'binning': [2, -0.5, 1.5],
        'bindata': {
            'data': [120.0, 180.0],
            'bkg': [100.0, 150.0],
            'sig': [30.0, 95.0]
        }
    }
    return source


@pytest.fixture(scope='module')
def spec_1bin_normsys(source=source_1bin_normsys()):
    spec = {
        'channels': [
            {
                'name': 'singlechannel',
                'samples': [
                    {
                        'name': 'signal',
                        'data': source['bindata']['sig'],
                        'modifiers': [
                            {
                                'name': 'mu',
                                'type': 'normfactor',
                                'data': None
                            }
                        ]
                    },
                    {
                        'name': 'background',
                        'data': source['bindata']['bkg'],
                        'modifiers': [
                            {
                                'name': 'bkg_norm',
                                'type': 'normsys',
                                'data': {'lo': 0.90, 'hi': 1.10}
                            }
                        ]
                    }
                ]
            }
        ]
    }
    return spec


@pytest.fixture(scope='module')
def expected_result_1bin_normsys(mu=1.):
    if mu == 1:
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
    return expected_result


@pytest.fixture(scope='module')
def setup_1bin_normsys(source=source_1bin_normsys(),
                       spec=spec_1bin_normsys(source_1bin_normsys()),
                       mu=1,
                       expected_result=expected_result_1bin_normsys(1.),
                       config={'init_pars': 2, 'par_bounds': 2}):
    return {
        'source': source,
        'spec': spec,
        'mu': mu,
        'expected': {
            'result': expected_result,
            'config': config
        }
    }


@pytest.fixture(scope='module')
def source_2bin_histosys_example2():
    return json.load(open('validation/data/2bin_histosys_example2.json'))


@pytest.fixture(scope='module')
def spec_2bin_histosys(source=source_2bin_histosys_example2()):
    spec = {
        'channels': [
            {
                'name': 'singlechannel',
                'samples': [
                    {
                        'name': 'signal',
                        'data': source['bindata']['sig'],
                        'modifiers': [
                            {
                                'name': 'mu',
                                'type': 'normfactor',
                                'data': None
                            }
                        ]
                    },
                    {
                        'name': 'background',
                        'data': source['bindata']['bkg'],
                        'modifiers': [
                            {
                                'name': 'bkg_norm',
                                'type': 'histosys',
                                'data': {
                                    'lo_data': source['bindata']['bkgsys_dn'],
                                    'hi_data': source['bindata']['bkgsys_up']
                                }
                            }
                        ]
                    }
                ]
            }
        ]
    }
    return spec


@pytest.fixture(scope='module')
def expected_result_2bin_histosys(mu=1):
    if mu == 1:
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
    return expected_result


@pytest.fixture(scope='module')
def setup_2bin_histosys(source=source_2bin_histosys_example2(),
                        spec=spec_2bin_histosys(
                            source_2bin_histosys_example2()),
                        mu=1,
                        expected_result=expected_result_2bin_histosys(1.),
                        config={'init_pars': 2, 'par_bounds': 2}):
    return {
        'source': source,
        'spec': spec,
        'mu': mu,
        'expected': {
            'result': expected_result,
            'config': config
        }
    }


@pytest.fixture(scope='module')
def source_2bin_2channel_example1():
    return json.load(open('validation/data/2bin_2channel_example1.json'))


@pytest.fixture(scope='module')
def spec_2bin_2channel(source=source_2bin_2channel_example1()):
    spec = {
        'channels': [
            {
                'name': 'signal',
                'samples': [
                    {
                        'name': 'signal',
                        'data': source['channels']['signal']['bindata']['sig'],
                        'modifiers': [
                            {
                                'name': 'mu',
                                'type': 'normfactor',
                                'data': None
                            }
                        ]
                    },
                    {
                        'name': 'background',
                        'data': source['channels']['signal']['bindata']['bkg'],
                        'modifiers': [
                            {
                                'name': 'uncorr_bkguncrt_signal',
                                'type': 'shapesys',
                                'data': source['channels']['signal']['bindata']['bkgerr']
                            }
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
                            {
                                'name': 'uncorr_bkguncrt_control',
                                'type': 'shapesys',
                                'data': source['channels']['control']['bindata']['bkgerr']
                            }
                        ]
                    }
                ]
            }
        ]
    }
    return spec


@pytest.fixture(scope='module')
def expected_result_2bin_2channel(mu=1.):
    if mu == 1:
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
    return expected_result


@pytest.fixture(scope='module')
def setup_2bin_2channel(source=source_2bin_2channel_example1(),
                        spec=spec_2bin_2channel(
                            source_2bin_2channel_example1()),
                        mu=1,
                        expected_result=expected_result_2bin_2channel(1.),
                        config={'init_pars': 5, 'par_bounds': 5}):
    # 1 mu + 2 gammas for 2 channels each
    return {
        'source': source,
        'spec': spec,
        'mu': mu,
        'expected': {
            'result': expected_result,
            'config': config
        }
    }


@pytest.fixture(scope='module')
def source_2bin_2channel_couplednorm():
    return json.load(open('validation/data/2bin_2channel_couplednorm.json'))


@pytest.fixture(scope='module')
def spec_2bin_2channel_couplednorm(source=source_2bin_2channel_couplednorm()):
    spec = {
        'channels': [
            {
                'name': 'signal',
                'samples': [
                    {
                        'name': 'signal',
                        'data': source['channels']['signal']['bindata']['sig'],
                        'modifiers': [
                            {
                                'name': 'mu',
                                'type': 'normfactor',
                                'data': None
                            }
                        ]
                    },
                    {
                        'name': 'bkg1',
                        'data': source['channels']['signal']['bindata']['bkg1'],
                        'modifiers': [
                            {
                                'name': 'coupled_normsys',
                                'type': 'normsys',
                                'data':  {'lo': 0.9, 'hi': 1.1}
                            }
                        ]
                    },
                    {
                        'name': 'bkg2',
                        'data': source['channels']['signal']['bindata']['bkg2'],
                        'modifiers': [
                            {
                                'name': 'coupled_normsys',
                                'type': 'normsys',
                                'data':  {'lo': 0.5, 'hi': 1.5}
                            }
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
                            {
                                'name': 'coupled_normsys',
                                'type': 'normsys',
                                'data': {'lo': 0.9, 'hi': 1.1}
                            }
                        ]
                    }
                ]
            }
        ]
    }
    return spec


@pytest.fixture(scope='module')
def expected_result_2bin_2channel_couplednorm(mu=1.):
    if mu == 1:
        expected_result = {
            'obs': 0.5999662863185762,
            'exp': [
                0.06596134134354742,
                0.15477912571478988,
                0.33323967895587736,
                0.6096429330789306,
                0.8688213053042003
            ]
        }
    return expected_result


@pytest.fixture(scope='module')
def setup_2bin_2channel_couplednorm(
        source=source_2bin_2channel_couplednorm(),
        spec=spec_2bin_2channel_couplednorm(
            source_2bin_2channel_couplednorm()),
        mu=1,
        expected_result=expected_result_2bin_2channel_couplednorm(1.),
        config={'init_pars': 2, 'par_bounds': 2}):
    # 1 mu + 1 alpha
    return {
        'source': source,
        'spec': spec,
        'mu': mu,
        'expected': {
            'result': expected_result,
            'config': config
        }
    }


@pytest.fixture(scope='module')
def source_2bin_2channel_coupledhisto():
    return json.load(open('validation/data/2bin_2channel_coupledhisto.json'))


@pytest.fixture(scope='module')
def spec_2bin_2channel_coupledhistosys(source=source_2bin_2channel_coupledhisto()):
    spec = {
        'channels': [
            {
                'name': 'signal',
                'samples': [
                    {
                        'name': 'signal',
                        'data': source['channels']['signal']['bindata']['sig'],
                        'modifiers': [
                            {
                                'name': 'mu',
                                'type': 'normfactor',
                                'data': None
                            }
                        ]
                    },
                    {
                        'name': 'bkg1',
                        'data': source['channels']['signal']['bindata']['bkg1'],
                        'modifiers': [
                            {
                                'name': 'coupled_histosys',
                                'type': 'histosys',
                                'data': {
                                    'lo_data': source['channels']['signal']['bindata']['bkg1_dn'],
                                    'hi_data': source['channels']['signal']['bindata']['bkg1_up']
                                }
                            }
                        ]
                    },
                    {
                        'name': 'bkg2',
                        'data': source['channels']['signal']['bindata']['bkg2'],
                        'modifiers': [
                            {
                                'name': 'coupled_histosys',
                                'type': 'histosys',
                                'data': {
                                    'lo_data': source['channels']['signal']['bindata']['bkg2_dn'],
                                    'hi_data': source['channels']['signal']['bindata']['bkg2_up']
                                }
                            }
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
                            {
                                'name': 'coupled_histosys',
                                'type': 'histosys',
                                'data': {
                                    'lo_data': source['channels']['control']['bindata']['bkg1_dn'],
                                    'hi_data': source['channels']['control']['bindata']['bkg1_up']
                                }
                            }
                        ]
                    }
                ]
            }
        ]
    }
    return spec


@pytest.fixture(scope='module')
def expected_result_2bin_2channel_coupledhistosys(mu=1.):
    if mu == 1:
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
    return expected_result


@pytest.fixture(scope='module')
def setup_2bin_2channel_coupledhistosys(
        source=source_2bin_2channel_coupledhisto(),
        spec=spec_2bin_2channel_coupledhistosys(
            source_2bin_2channel_coupledhisto()),
        mu=1,
        expected_result=expected_result_2bin_2channel_coupledhistosys(1.),
        config={'auxdata': 1, 'init_pars': 2, 'par_bounds': 2}):
    # 1 mu 1 shared histosys
    return {
        'source': source,
        'spec': spec,
        'mu': mu,
        'expected': {
            'result': expected_result,
            'config': config
        }
    }


@pytest.fixture(scope='module')
def source_2bin_2channel_coupledshapefactor():
    return json.load(open('validation/data/2bin_2channel_coupledshapefactor.json'))


@pytest.fixture(scope='module')
def spec_2bin_2channel_coupledshapefactor(source=source_2bin_2channel_coupledshapefactor()):
    spec = {
        'channels': [
            {
                'name': 'signal',
                'samples': [
                    {
                        'name': 'signal',
                        'data': source['channels']['signal']['bindata']['sig'],
                        'modifiers': [
                            {
                                'name': 'mu',
                                'type': 'normfactor',
                                'data': None
                            }
                        ]
                    },
                    {
                        'name': 'bkg1',
                        'data': source['channels']['signal']['bindata']['bkg1'],
                        'modifiers': [
                            {
                                'name': 'coupled_shapefactor',
                                'type': 'shapefactor',
                                'data': None
                            }
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
                            {
                                'name': 'coupled_shapefactor',
                                'type': 'shapefactor',
                                'data': None
                            }
                        ]
                    }
                ]
            }
        ]
    }
    return spec


@pytest.fixture(scope='module')
def expected_result_2bin_2channel_coupledshapefactor(mu=1.):
    if mu == 1:
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
    return expected_result


@pytest.fixture(scope='module')
def setup_2bin_2channel_coupledshapefactor(
        source=source_2bin_2channel_coupledshapefactor(),
        spec=spec_2bin_2channel_coupledshapefactor(
            source_2bin_2channel_coupledshapefactor()),
        mu=1,
        expected_result=expected_result_2bin_2channel_coupledshapefactor(1.),
        config={'auxdata': 0, 'init_pars': 3, 'par_bounds': 3}):
    # 1 mu 2 shared shapefactors
    return {
        'source': source,
        'spec': spec,
        'mu': mu,
        'expected': {
            'result': expected_result,
            'config': config
        }
    }


def validate_runOnePoint(pdf, data, mu_test, expected_result, tolerance=1e-5):
    init_pars = pdf.config.suggested_init()
    par_bounds = pdf.config.suggested_bounds()

    CLs_obs, CLs_exp = pyhf.utils.runOnePoint(
        mu_test, data, pdf, init_pars, par_bounds)[-2:]
    assert (CLs_obs - expected_result['obs']) / \
        expected_result['obs'] < tolerance
    for result, expected_result in zip(CLs_exp, expected_result['exp']):
        assert (result - expected_result) / \
            expected_result < tolerance


@pytest.mark.parametrize('setup', [
    setup_1bin_shapesys(),
    setup_1bin_normsys(),
    setup_2bin_histosys(),
    setup_2bin_2channel(),
    setup_2bin_2channel_couplednorm(),
    setup_2bin_2channel_coupledhistosys(),
    setup_2bin_2channel_coupledshapefactor()
],
    ids=[
    '1bin_shapesys_mu1',
    '1bin_normsys_mu1',
    '2bin_histosys_mu1',
    '2bin_2channel_mu1',
    '2bin_2channel_couplednorm_mu1',
    '2bin_2channel_coupledhistosys_mu1',
    '2bin_2channel_coupledshapefactor_mu1'
])
def test_validation(setup):
    source = setup['source']
    pdf = pyhf.Model(setup['spec'])

    if 'channels' in source:
        data = []
        for c in pdf.spec['channels']:
            data += source['channels'][c['name']]['bindata']['data']
        data = data + pdf.config.auxdata
    else:
        data = source['bindata']['data'] + pdf.config.auxdata

    if 'auxdata' in setup['expected']['config']:
        assert len(pdf.config.auxdata) == \
            setup['expected']['config']['auxdata']
    assert len(pdf.config.suggested_init()) == \
        setup['expected']['config']['init_pars']
    assert len(pdf.config.suggested_bounds()) == \
        setup['expected']['config']['par_bounds']

    validate_runOnePoint(pdf, data, setup['mu'], setup['expected']['result'])


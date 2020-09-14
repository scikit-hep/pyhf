import pyhf
import pyhf.writexml
import pyhf.readxml
import json
import pytest
from pathlib import Path
import numpy as np


@pytest.fixture(scope='module')
def source_1bin_shapesys():
    with open('validation/data/1bin_example1.json') as read_json:
        return json.load(read_json)


@pytest.fixture(scope='module')
def spec_1bin_shapesys(source_1bin_shapesys):
    source = source_1bin_shapesys
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
                        ],
                    },
                    {
                        'name': 'background',
                        'data': source['bindata']['bkg'],
                        'modifiers': [
                            {
                                'name': 'uncorr_bkguncrt',
                                'type': 'shapesys',
                                'data': source['bindata']['bkgerr'],
                            }
                        ],
                    },
                ],
            }
        ]
    }
    return spec


@pytest.fixture(scope='module')
def expected_result_1bin_shapesys():
    expected_result = {
        "exp": [
            0.06372011644331387,
            0.1509686618126131,
            0.3279657430196915,
            0.604613569829645,
            0.8662652332047568,
        ],
        "obs": 0.45418892944576333,
    }
    return expected_result


@pytest.fixture(scope='module')
def source_1bin_lumi():
    with open('validation/data/1bin_lumi.json') as read_json:
        return json.load(read_json)


@pytest.fixture(scope='module')
def spec_1bin_lumi(source_1bin_lumi):
    source = source_1bin_lumi
    spec = {
        "channels": [
            {
                "name": "channel1",
                "samples": [
                    {
                        "name": "signal",
                        "data": source['bindata']['sig'],
                        "modifiers": [
                            {"data": None, "name": "mu", "type": "normfactor"}
                        ],
                    },
                    {
                        "name": "background1",
                        "data": source['bindata']['bkg1'],
                        "modifiers": [{"data": None, "name": "lumi", "type": "lumi"}],
                    },
                    {
                        "name": "background2",
                        "data": source['bindata']['bkg2'],
                        "modifiers": [{"data": None, "name": "lumi", "type": "lumi"}],
                    },
                ],
            }
        ],
        "parameters": [
            {
                "auxdata": [1.0],
                "bounds": [[0.0, 10.0]],
                "inits": [1.0],
                "name": "lumi",
                "sigmas": [0.1],
            }
        ],
    }
    return spec


@pytest.fixture(scope='module')
def expected_result_1bin_lumi():
    expected_result = {
        "exp": [
            0.01060400765567206,
            0.04022451457730529,
            0.13614632580079802,
            0.37078985531427255,
            0.7110468540175344,
        ],
        "obs": 0.010473144401519705,
    }
    return expected_result


@pytest.fixture(scope='module')
def source_1bin_normsys():
    with open('validation/data/1bin_normsys.json') as read_json:
        return json.load(read_json)


@pytest.fixture(scope='module')
def spec_1bin_normsys(source_1bin_normsys):
    source = source_1bin_normsys
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
                        ],
                    },
                    {
                        'name': 'background',
                        'data': source['bindata']['bkg'],
                        'modifiers': [
                            {
                                'name': 'bkg_norm',
                                'type': 'normsys',
                                'data': {'lo': 0.90, 'hi': 1.10},
                            }
                        ],
                    },
                ],
            }
        ]
    }
    return spec


@pytest.fixture(scope='module')
def expected_result_1bin_normsys():
    expected_result = {
        "exp": [
            7.472581399417304e-10,
            5.741738272450336e-08,
            3.690120950161796e-06,
            0.00016966882793076826,
            0.004392935288879465,
        ],
        "obs": 0.0006735336290569807,
    }
    return expected_result


@pytest.fixture(scope='module')
def source_2bin_histosys():
    with open('validation/data/2bin_histosys_example2.json') as read_json:
        return json.load(read_json)


@pytest.fixture(scope='module')
def spec_2bin_histosys(source_2bin_histosys):
    source = source_2bin_histosys
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
                        ],
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
                                    'hi_data': source['bindata']['bkgsys_up'],
                                },
                            }
                        ],
                    },
                ],
            }
        ]
    }
    return spec


@pytest.fixture(scope='module')
def expected_result_2bin_histosys():
    expected_result = {
        "exp": [
            7.133904244038431e-06,
            0.00012547100627138575,
            0.001880010666437615,
            0.02078964907605385,
            0.13692494523572218,
        ],
        "obs": 0.1001463460725534,
    }
    return expected_result


@pytest.fixture(scope='module')
def source_2bin_2channel():
    with open('validation/data/2bin_2channel_example1.json') as read_json:
        return json.load(read_json)


@pytest.fixture(scope='module')
def spec_2bin_2channel(source_2bin_2channel):
    source = source_2bin_2channel
    spec = {
        'channels': [
            {
                'name': 'signal',
                'samples': [
                    {
                        'name': 'signal',
                        'data': source['channels']['signal']['bindata']['sig'],
                        'modifiers': [
                            {'name': 'mu', 'type': 'normfactor', 'data': None}
                        ],
                    },
                    {
                        'name': 'background',
                        'data': source['channels']['signal']['bindata']['bkg'],
                        'modifiers': [
                            {
                                'name': 'uncorr_bkguncrt_signal',
                                'type': 'shapesys',
                                'data': source['channels']['signal']['bindata'][
                                    'bkgerr'
                                ],
                            }
                        ],
                    },
                ],
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
                                'data': source['channels']['control']['bindata'][
                                    'bkgerr'
                                ],
                            }
                        ],
                    }
                ],
            },
        ]
    }
    return spec


@pytest.fixture(scope='module')
def expected_result_2bin_2channel():
    expected_result = {
        "exp": [
            0.0004349234603527283,
            0.003422361539161119,
            0.02337454317608372,
            0.12186650297311125,
            0.40382274594391104,
        ],
        "obs": 0.0563327694384318,
    }
    return expected_result


@pytest.fixture(scope='module')
def source_2bin_2channel_couplednorm():
    with open('validation/data/2bin_2channel_couplednorm.json') as read_json:
        return json.load(read_json)


@pytest.fixture(scope='module')
def spec_2bin_2channel_couplednorm(source_2bin_2channel_couplednorm):
    source = source_2bin_2channel_couplednorm
    spec = {
        'channels': [
            {
                'name': 'signal',
                'samples': [
                    {
                        'name': 'signal',
                        'data': source['channels']['signal']['bindata']['sig'],
                        'modifiers': [
                            {'name': 'mu', 'type': 'normfactor', 'data': None}
                        ],
                    },
                    {
                        'name': 'bkg1',
                        'data': source['channels']['signal']['bindata']['bkg1'],
                        'modifiers': [
                            {
                                'name': 'coupled_normsys',
                                'type': 'normsys',
                                'data': {'lo': 0.9, 'hi': 1.1},
                            }
                        ],
                    },
                    {
                        'name': 'bkg2',
                        'data': source['channels']['signal']['bindata']['bkg2'],
                        'modifiers': [
                            {
                                'name': 'coupled_normsys',
                                'type': 'normsys',
                                'data': {'lo': 0.5, 'hi': 1.5},
                            }
                        ],
                    },
                ],
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
                                'data': {'lo': 0.9, 'hi': 1.1},
                            }
                        ],
                    }
                ],
            },
        ]
    }
    return spec


@pytest.fixture(scope='module')
def expected_result_2bin_2channel_couplednorm():
    # NB: mac/linux differ for exp[0]
    # Mac:   0.055222676184648795
    # Linux: 0.05522273289103311
    # Fill with midpoint of both values
    expected_result = {
        "exp": [
            0.05522270453784095,
            0.1361301880753241,
            0.30686879632329855,
            0.5839437910061168,
            0.8554708284963864,
        ],
        "obs": 0.5906216823766879,
    }
    return expected_result


@pytest.fixture(scope='module')
def source_2bin_2channel_coupledhistosys():
    with open('validation/data/2bin_2channel_coupledhisto.json') as read_json:
        return json.load(read_json)


@pytest.fixture(scope='module')
def spec_2bin_2channel_coupledhistosys(source_2bin_2channel_coupledhistosys):
    source = source_2bin_2channel_coupledhistosys
    spec = {
        'channels': [
            {
                'name': 'signal',
                'samples': [
                    {
                        'name': 'signal',
                        'data': source['channels']['signal']['bindata']['sig'],
                        'modifiers': [
                            {'name': 'mu', 'type': 'normfactor', 'data': None}
                        ],
                    },
                    {
                        'name': 'bkg1',
                        'data': source['channels']['signal']['bindata']['bkg1'],
                        'modifiers': [
                            {
                                'name': 'coupled_histosys',
                                'type': 'histosys',
                                'data': {
                                    'lo_data': source['channels']['signal']['bindata'][
                                        'bkg1_dn'
                                    ],
                                    'hi_data': source['channels']['signal']['bindata'][
                                        'bkg1_up'
                                    ],
                                },
                            }
                        ],
                    },
                    {
                        'name': 'bkg2',
                        'data': source['channels']['signal']['bindata']['bkg2'],
                        'modifiers': [
                            {
                                'name': 'coupled_histosys',
                                'type': 'histosys',
                                'data': {
                                    'lo_data': source['channels']['signal']['bindata'][
                                        'bkg2_dn'
                                    ],
                                    'hi_data': source['channels']['signal']['bindata'][
                                        'bkg2_up'
                                    ],
                                },
                            }
                        ],
                    },
                ],
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
                                    'lo_data': source['channels']['control']['bindata'][
                                        'bkg1_dn'
                                    ],
                                    'hi_data': source['channels']['control']['bindata'][
                                        'bkg1_up'
                                    ],
                                },
                            }
                        ],
                    }
                ],
            },
        ]
    }
    return spec


@pytest.fixture(scope='module')
def expected_result_2bin_2channel_coupledhistosys():
    expected_result = {
        "exp": [
            1.7654378902209275e-05,
            0.00026266409358853543,
            0.0033401113778672156,
            0.03152286332324451,
            0.17907927340107824,
        ],
        "obs": 0.07967400132261188,
    }
    return expected_result


@pytest.fixture(scope='module')
def source_2bin_2channel_coupledshapefactor():
    with open('validation/data/2bin_2channel_coupledshapefactor.json') as read_json:
        return json.load(read_json)


@pytest.fixture(scope='module')
def spec_2bin_2channel_coupledshapefactor(source_2bin_2channel_coupledshapefactor):
    source = source_2bin_2channel_coupledshapefactor
    spec = {
        'channels': [
            {
                'name': 'signal',
                'samples': [
                    {
                        'name': 'signal',
                        'data': source['channels']['signal']['bindata']['sig'],
                        'modifiers': [
                            {'name': 'mu', 'type': 'normfactor', 'data': None}
                        ],
                    },
                    {
                        'name': 'bkg1',
                        'data': source['channels']['signal']['bindata']['bkg1'],
                        'modifiers': [
                            {
                                'name': 'coupled_shapefactor',
                                'type': 'shapefactor',
                                'data': None,
                            }
                        ],
                    },
                ],
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
                                'data': None,
                            }
                        ],
                    }
                ],
            },
        ]
    }
    return spec


@pytest.fixture(scope='module')
def expected_result_2bin_2channel_coupledshapefactor():
    expected_result = {
        'obs': 0.5421679124909312,
        'exp': [
            0.013753299929451691,
            0.048887400056355966,
            0.15555296253957684,
            0.4007561343326305,
            0.7357169630955912,
        ],
    }
    return expected_result


def validate_hypotest(pdf, data, mu_test, expected_result, tolerance=1e-6):
    init_pars = pdf.config.suggested_init()
    par_bounds = pdf.config.suggested_bounds()

    CLs_obs, CLs_exp_set = pyhf.infer.hypotest(
        mu_test,
        data,
        pdf,
        init_pars,
        par_bounds,
        return_expected_set=True,
        qtilde=False,
    )
    assert abs(CLs_obs - expected_result['obs']) / expected_result['obs'] < tolerance
    for result, expected in zip(CLs_exp_set, expected_result['exp']):
        assert abs(result - expected) / expected < tolerance, result


@pytest.fixture(
    params=[
        ('1bin_shapesys', {'init_pars': 2, 'par_bounds': 2}, 1e-6),
        ('1bin_lumi', {'init_pars': 2, 'par_bounds': 2}, 4e-6),
        ('1bin_normsys', {'init_pars': 2, 'par_bounds': 2}, 2e-9),
        ('2bin_histosys', {'init_pars': 2, 'par_bounds': 2}, 8e-5),
        ('2bin_2channel', {'init_pars': 5, 'par_bounds': 5}, 1e-6),
        ('2bin_2channel_couplednorm', {'init_pars': 2, 'par_bounds': 2}, 1e-6),
        (
            '2bin_2channel_coupledhistosys',
            {'auxdata': 1, 'init_pars': 2, 'par_bounds': 2},
            1e-6,
        ),
        (
            '2bin_2channel_coupledshapefactor',
            {'auxdata': 0, 'init_pars': 3, 'par_bounds': 3},
            2.5e-6,
        ),
    ],
    ids=[
        '1bin_shapesys_mu1',
        '1bin_lumi_mu1',
        '1bin_normsys_mu1',
        '2bin_histosys_mu1',
        '2bin_2channel_mu1',
        '2bin_2channel_couplednorm_mu1',
        '2bin_2channel_coupledhistosys_mu1',
        '2bin_2channel_coupledshapefactor_mu1',
    ],
)
def setup_and_tolerance(request):
    _name = request.param[0]
    source = request.getfixturevalue(f"source_{_name}")
    spec = request.getfixturevalue(f"spec_{_name}")
    expected_result = request.getfixturevalue(f"expected_result_{_name}")
    config = request.param[1]
    tolerance = request.param[2]
    return (
        {
            'source': source,
            'spec': spec,
            'mu': 1.0,
            'expected': {'result': expected_result, 'config': config},
        },
        tolerance,
    )


def test_validation(setup_and_tolerance):
    setup, tolerance = setup_and_tolerance
    source = setup['source']

    pdf = pyhf.Model(setup['spec'])

    if 'channels' in source:
        data = []
        for c in pdf.config.channels:
            data += source['channels'][c]['bindata']['data']
        data = data + pdf.config.auxdata
    else:
        data = source['bindata']['data'] + pdf.config.auxdata

    if 'auxdata' in setup['expected']['config']:
        assert len(pdf.config.auxdata) == setup['expected']['config']['auxdata']
    assert len(pdf.config.suggested_init()) == setup['expected']['config']['init_pars']
    assert (
        len(pdf.config.suggested_bounds()) == setup['expected']['config']['par_bounds']
    )

    validate_hypotest(
        pdf, data, setup['mu'], setup['expected']['result'], tolerance=tolerance
    )


@pytest.mark.parametrize(
    'toplvl, basedir',
    [
        (
            'validation/xmlimport_input/config/example.xml',
            'validation/xmlimport_input/',
        ),
        (
            'validation/xmlimport_input2/config/example.xml',
            'validation/xmlimport_input2',
        ),
        (
            'validation/xmlimport_input3/config/examples/example_ShapeSys.xml',
            'validation/xmlimport_input3',
        ),
    ],
    ids=['example-one', 'example-two', 'example-three'],
)
def test_import_roundtrip(tmpdir, toplvl, basedir):
    parsed_xml_before = pyhf.readxml.parse(toplvl, basedir)
    spec = {
        'channels': parsed_xml_before['channels'],
        'parameters': parsed_xml_before['measurements'][0]['config']['parameters'],
    }
    pdf_before = pyhf.Model(spec, poi_name='SigXsecOverSM')

    tmpconfig = tmpdir.mkdir('config')
    tmpdata = tmpdir.mkdir('data')
    tmpxml = tmpdir.join('FitConfig.xml')
    tmpxml.write(
        pyhf.writexml.writexml(
            parsed_xml_before,
            tmpconfig.strpath,
            tmpdata.strpath,
            Path(tmpdir.strpath).joinpath('FitConfig'),
        ).decode('utf-8')
    )
    parsed_xml_after = pyhf.readxml.parse(tmpxml.strpath, tmpdir.strpath)
    spec = {
        'channels': parsed_xml_after['channels'],
        'parameters': parsed_xml_after['measurements'][0]['config']['parameters'],
    }
    pdf_after = pyhf.Model(spec, poi_name='SigXsecOverSM')

    data_before = [
        binvalue
        for k in pdf_before.config.channels
        for binvalue in next(
            obs for obs in parsed_xml_before['observations'] if obs['name'] == k
        )['data']
    ] + pdf_before.config.auxdata

    data_after = [
        binvalue
        for k in pdf_after.config.channels
        for binvalue in next(
            obs for obs in parsed_xml_after['observations'] if obs['name'] == k
        )['data']
    ] + pdf_after.config.auxdata

    assert data_before == data_after

    init_pars_before = pdf_before.config.suggested_init()
    init_pars_after = pdf_after.config.suggested_init()
    assert init_pars_before == init_pars_after

    par_bounds_before = pdf_before.config.suggested_bounds()
    par_bounds_after = pdf_after.config.suggested_bounds()
    assert par_bounds_before == par_bounds_after

    CLs_obs_before, CLs_exp_set_before = pyhf.infer.hypotest(
        1,
        data_before,
        pdf_before,
        init_pars_before,
        par_bounds_before,
        return_expected_set=True,
    )
    CLs_obs_after, CLs_exp_set_after = pyhf.infer.hypotest(
        1,
        data_after,
        pdf_after,
        init_pars_after,
        par_bounds_after,
        return_expected_set=True,
    )

    tolerance = 1e-6
    assert abs(CLs_obs_after - CLs_obs_before) / CLs_obs_before < tolerance
    for result, expected_result in zip(CLs_exp_set_after, CLs_exp_set_before):
        assert abs(result - expected_result) / expected_result < tolerance


def test_shapesys_nuisparfilter_validation():
    reference_root_results = {
        "CLs_exp": [
            2.702197937866914e-05,
            0.00037099917612576155,
            0.004360634386335687,
            0.03815031509701916,
            0.20203027564155074,
        ],
        "CLs_obs": 0.004360634405484502,
    }
    null = None
    spec = {
        "channels": [
            {
                "name": "channel1",
                "samples": [
                    {
                        "data": [20, 10],
                        "modifiers": [
                            {
                                "data": null,
                                "name": "SigXsecOverSM",
                                "type": "normfactor",
                            }
                        ],
                        "name": "signal",
                    },
                    {
                        "data": [100, 10],
                        "modifiers": [
                            {"data": [10, 0], "name": "syst", "type": "shapesys"}
                        ],
                        "name": "background1",
                    },
                ],
            }
        ],
        "measurements": [
            {
                "config": {
                    "parameters": [
                        {
                            "auxdata": [1],
                            "bounds": [[0.5, 1.5]],
                            "inits": [1],
                            "name": "lumi",
                            "sigmas": [0.1],
                        }
                    ],
                    "poi": "SigXsecOverSM",
                },
                "name": "GaussExample",
            }
        ],
        "observations": [{"data": [100, 10], "name": "channel1"}],
        "version": "1.0.0",
    }
    ws = pyhf.Workspace(spec)
    model = ws.model(
        modifier_settings={
            'normsys': {'interpcode': 'code4'},
            'histosys': {'interpcode': 'code4p'},
        },
    )
    data = ws.data(model)
    obs, exp = pyhf.infer.hypotest(1.0, data, model, return_expected_set=True)
    pyhf_results = {'CLs_obs': obs, 'CLs_exp': [e for e in exp]}

    assert np.allclose(
        reference_root_results['CLs_obs'], pyhf_results['CLs_obs'], atol=1e-4, rtol=1e-5
    )
    assert np.allclose(
        reference_root_results['CLs_exp'], pyhf_results['CLs_exp'], atol=1e-4, rtol=1e-5
    )


@pytest.mark.parametrize(
    'backend',
    [
        pyhf.tensor.numpy_backend,
        pyhf.tensor.jax_backend,
        pyhf.tensor.tensorflow_backend,
        pyhf.tensor.pytorch_backend,
    ],
)
@pytest.mark.parametrize('optimizer', ['scipy', 'minuit'])
def test_optimizer_stitching(backend, optimizer):
    pyhf.set_backend(backend(precision='64b'), optimizer)

    pdf = pyhf.simplemodels.hepdata_like([50.0], [100.0], [10])
    data = [125.0] + pdf.config.auxdata

    result_nostitch = pyhf.infer.mle.fixed_poi_fit(2.0, data, pdf, do_stitch=False)
    result_stitch = pyhf.infer.mle.fixed_poi_fit(2.0, data, pdf, do_stitch=True)

    assert np.allclose(
        pyhf.tensorlib.tolist(result_nostitch),
        pyhf.tensorlib.tolist(result_stitch),
        rtol=4e-05,
    )


@pytest.mark.parametrize(
    'backend',
    [
        pyhf.tensor.jax_backend,
        pyhf.tensor.tensorflow_backend,
        pyhf.tensor.pytorch_backend,
    ],
)
@pytest.mark.parametrize('optimizer,rtol', [('scipy', 1e-6), ('minuit', 1e-3)])
def test_optimizer_grad(backend, optimizer, rtol):
    pyhf.set_backend(backend(precision='64b'), optimizer)

    pdf = pyhf.simplemodels.hepdata_like([50.0], [100.0], [10])
    data = [125.0] + pdf.config.auxdata

    result_nograd = pyhf.infer.mle.fit(data, pdf, do_grad=False)
    result_grad = pyhf.infer.mle.fit(data, pdf, do_grad=True)

    # TODO: let's make this agreement better
    assert np.allclose(
        pyhf.tensorlib.tolist(result_nograd),
        pyhf.tensorlib.tolist(result_grad),
        rtol=rtol,
        atol=1e-6,
    )

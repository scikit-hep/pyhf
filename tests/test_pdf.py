import pyhf
import pytest
import pyhf.exceptions
import numpy as np
import json


def test_minimum_model_spec():
    spec = {
        'channels': [
            {
                'name': 'channel',
                'samples': [
                    {
                        'name': 'goodsample',
                        'data': [1.0],
                        'modifiers': [
                            {'type': 'normfactor', 'name': 'mu', 'data': None}
                        ],
                    },
                ],
            }
        ]
    }
    pyhf.Model(spec)


def test_pdf_inputs(backend):
    source = {
        "binning": [2, -0.5, 1.5],
        "bindata": {"data": [55.0], "bkg": [50.0], "bkgerr": [7.0], "sig": [10.0]},
    }
    pdf = pyhf.simplemodels.uncorrelated_background(
        source['bindata']['sig'], source['bindata']['bkg'], source['bindata']['bkgerr']
    )

    pars = pdf.config.suggested_init()
    data = source['bindata']['data'] + pdf.config.auxdata

    tensorlib, _ = backend
    assert tensorlib.shape(tensorlib.astensor(data)) == (2,)
    assert tensorlib.shape(tensorlib.astensor(pars)) == (2,)
    assert tensorlib.tolist(pdf.pdf(pars, data)) == pytest.approx(
        [0.002417160663753748], abs=1e-4
    )
    assert tensorlib.tolist(pdf.logpdf(pars, data)) == pytest.approx(
        [-6.025179228209936], abs=1e-4
    )


def test_invalid_pdf_pars():
    source = {
        "binning": [2, -0.5, 1.5],
        "bindata": {"data": [55.0], "bkg": [50.0], "bkgerr": [7.0], "sig": [10.0]},
    }
    pdf = pyhf.simplemodels.uncorrelated_background(
        source['bindata']['sig'], source['bindata']['bkg'], source['bindata']['bkgerr']
    )

    pars = pdf.config.suggested_init() + [1.0]
    data = source['bindata']['data'] + pdf.config.auxdata

    with pytest.raises(pyhf.exceptions.InvalidPdfParameters):
        pdf.logpdf(pars, data)


def test_invalid_pdf_data():
    source = {
        "binning": [2, -0.5, 1.5],
        "bindata": {"data": [55.0], "bkg": [50.0], "bkgerr": [7.0], "sig": [10.0]},
    }
    pdf = pyhf.simplemodels.uncorrelated_background(
        source['bindata']['sig'], source['bindata']['bkg'], source['bindata']['bkgerr']
    )

    pars = pdf.config.suggested_init()
    data = source['bindata']['data'] + [10.0] + pdf.config.auxdata

    with pytest.raises(pyhf.exceptions.InvalidPdfData):
        pdf.logpdf(pars, data)


@pytest.mark.parametrize('batch_size', [None, 2])
def test_pdf_expected_data_by_sample(backend, batch_size):
    tb, _ = backend
    source = {
        "binning": [2, -0.5, 1.5],
        "bindata": {"data": [55.0], "bkg": [50.0], "bkgerr": [7.0], "sig": [10.0]},
    }
    pdf = pyhf.simplemodels.uncorrelated_background(
        source['bindata']['sig'],
        source['bindata']['bkg'],
        source['bindata']['bkgerr'],
        batch_size=batch_size,
    )

    nrepeats = (batch_size, 1) if batch_size else (1,)
    init_pars = tb.tile(tb.astensor(pdf.config.suggested_init()), nrepeats)
    expected_data = tb.tile(tb.astensor([60]), nrepeats)
    expected_bkg = tb.tile(tb.astensor([50]), nrepeats)
    expected_sig = tb.tile(tb.astensor([10]), nrepeats)

    assert tb.tolist(pdf.main_model.expected_data(init_pars)) == tb.tolist(
        expected_data
    )

    data = pdf.main_model.expected_data(init_pars, return_by_sample=True)
    if batch_size:
        data = tb.tolist(tb.einsum('ij...->ji...', data))

    sample_expected_data = dict(zip(pdf.config.samples, tb.tolist(data)))
    assert sample_expected_data['background'] == tb.tolist(expected_bkg)
    assert sample_expected_data['signal'] == tb.tolist(expected_sig)


def test_pdf_basicapi_tests(backend):
    source = {
        "binning": [2, -0.5, 1.5],
        "bindata": {"data": [55.0], "bkg": [50.0], "bkgerr": [7.0], "sig": [10.0]},
    }
    pdf = pyhf.simplemodels.uncorrelated_background(
        source['bindata']['sig'], source['bindata']['bkg'], source['bindata']['bkgerr']
    )

    pars = pdf.config.suggested_init()
    data = source['bindata']['data'] + pdf.config.auxdata

    tensorlib, _ = backend
    assert tensorlib.tolist(pdf.pdf(pars, data)) == pytest.approx(
        [0.002417118312751542], 2.5e-05
    )
    assert tensorlib.tolist(pdf.expected_data(pars)) == pytest.approx(
        [60.0, 51.020408630], 1e-08
    )

    assert tensorlib.tolist(pdf.expected_actualdata(pars)) == pytest.approx(
        [60.0], 1e-08
    )
    assert tensorlib.tolist(pdf.expected_auxdata(pars)) == pytest.approx(
        [51.020408630], 1e-08
    )
    assert tensorlib.tolist(pdf.main_model.expected_data(pars)) == pytest.approx(
        [60.0], 1e-08
    )

    pdf = pyhf.simplemodels.uncorrelated_background(
        source['bindata']['sig'],
        source['bindata']['bkg'],
        source['bindata']['bkgerr'],
        batch_size=2,
    )

    pars = [pdf.config.suggested_init()] * 2
    data = source['bindata']['data'] + pdf.config.auxdata

    tensorlib, _ = backend
    assert tensorlib.tolist(pdf.pdf(pars, data)) == pytest.approx(
        [0.002417118312751542] * 2, 2.5e-05
    )
    assert tensorlib.tolist(pdf.expected_data(pars))
    assert tensorlib.tolist(pdf.expected_data(pars)[0]) == pytest.approx(
        [60.0, 51.020408630], 1e-08
    )
    assert tensorlib.tolist(pdf.expected_data(pars)[1]) == pytest.approx(
        [60.0, 51.020408630], 1e-08
    )


@pytest.mark.only_numpy
def test_core_pdf_broadcasting(backend):
    data = [10, 11, 12, 13, 14, 15]
    lambdas = [15, 14, 13, 12, 11, 10]
    naive_python = [pyhf.tensorlib.poisson(d, lam) for d, lam in zip(data, lambdas)]

    broadcasted = pyhf.tensorlib.poisson(data, lambdas)

    assert np.array(data).shape == np.array(lambdas).shape
    assert broadcasted.shape == np.array(data).shape
    assert np.all(naive_python == broadcasted)

    data = [10, 11, 12, 13, 14, 15]
    mus = [15, 14, 13, 12, 11, 10]
    sigmas = [1, 2, 3, 4, 5, 6]
    naive_python = [
        pyhf.tensorlib.normal(d, mu, sig) for d, mu, sig in zip(data, mus, sigmas)
    ]

    broadcasted = pyhf.tensorlib.normal(data, mus, sigmas)

    assert np.array(data).shape == np.array(mus).shape
    assert np.array(data).shape == np.array(sigmas).shape
    assert broadcasted.shape == np.array(data).shape
    assert np.all(naive_python == broadcasted)


def test_pdf_integration_staterror(backend):
    spec = {
        'channels': [
            {
                'name': 'firstchannel',
                'samples': [
                    {
                        'name': 'mu',
                        'data': [10.0, 10.0],
                        'modifiers': [
                            {'name': 'mu', 'type': 'normfactor', 'data': None}
                        ],
                    },
                    {
                        'name': 'bkg1',
                        'data': [50.0, 70.0],
                        'modifiers': [
                            {
                                'name': 'stat_firstchannel',
                                'type': 'staterror',
                                'data': [12.0, 12.0],
                            }
                        ],
                    },
                    {
                        'name': 'bkg2',
                        'data': [30.0, 20.0],
                        'modifiers': [
                            {
                                'name': 'stat_firstchannel',
                                'type': 'staterror',
                                'data': [5.0, 5.0],
                            }
                        ],
                    },
                    {'name': 'bkg3', 'data': [20.0, 15.0], 'modifiers': []},
                ],
            }
        ]
    }
    pdf = pyhf.Model(spec)
    par_set = pdf.config.param_set('stat_firstchannel')
    tensorlib, _ = backend
    uncerts = tensorlib.astensor([[12.0, 12.0], [5.0, 5.0]])
    nominal = tensorlib.astensor([[50.0, 70.0], [30.0, 20.0]])
    quad = tensorlib.sqrt(tensorlib.sum(tensorlib.power(uncerts, 2), axis=0))
    totals = tensorlib.sum(nominal, axis=0)
    assert pytest.approx(tensorlib.tolist(par_set.sigmas)) == tensorlib.tolist(
        tensorlib.divide(quad, totals)
    )


def test_poiless_model(backend):
    spec = {
        'channels': [
            {
                'name': 'channel',
                'samples': [
                    {
                        'name': 'goodsample',
                        'data': [10.0],
                        'modifiers': [
                            {
                                'type': 'normsys',
                                'name': 'shape',
                                'data': {"hi": 0.5, "lo": 1.5},
                            }
                        ],
                    },
                ],
            }
        ]
    }
    model = pyhf.Model(spec, poi_name=None)

    data = [12] + model.config.auxdata
    pyhf.infer.mle.fit(data, model)

    with pytest.raises(pyhf.exceptions.UnspecifiedPOI):
        pyhf.infer.mle.fixed_poi_fit(1.0, data, model)

    with pytest.raises(pyhf.exceptions.UnspecifiedPOI):
        pyhf.infer.hypotest(1.0, data, model)


def test_poiless_model_empty_string(backend):
    spec = {
        'channels': [
            {
                'name': 'channel',
                'samples': [
                    {
                        'name': 'goodsample',
                        'data': [10.0],
                        'modifiers': [
                            {
                                'type': 'normsys',
                                'name': 'shape',
                                'data': {"hi": 0.5, "lo": 1.5},
                            }
                        ],
                    },
                ],
            }
        ]
    }
    model = pyhf.Model(spec, poi_name="")

    data = [12] + model.config.auxdata
    pyhf.infer.mle.fit(data, model)

    with pytest.raises(pyhf.exceptions.UnspecifiedPOI):
        pyhf.infer.mle.fixed_poi_fit(1.0, data, model)

    with pytest.raises(pyhf.exceptions.UnspecifiedPOI):
        pyhf.infer.hypotest(1.0, data, model)


def test_pdf_integration_shapesys_zeros(backend):
    spec = {
        "channels": [
            {
                "name": "channel1",
                "samples": [
                    {
                        "data": [20.0, 10.0, 5.0, 3.0, 2.0, 1.0],
                        "modifiers": [
                            {"data": None, "name": "mu", "type": "normfactor"}
                        ],
                        "name": "signal",
                    },
                    {
                        "data": [100.0, 90, 0.0, 70, 0.1, 50],
                        "modifiers": [
                            {
                                "data": [10, 9, 1, 0.0, 0.1, 5],
                                "name": "syst",
                                "type": "shapesys",
                            }
                        ],
                        "name": "background1",
                    },
                ],
            }
        ]
    }
    pdf = pyhf.Model(spec)
    par_set_syst = pdf.config.param_set('syst')

    assert par_set_syst.n_parameters == 6
    tensorlib, _ = backend
    nominal_sq = tensorlib.power(tensorlib.astensor([100.0, 90, 1.0, 1.0, 0.1, 50]), 2)
    uncerts_sq = tensorlib.power(tensorlib.astensor([10, 9, 1.0, 1.0, 0.1, 5]), 2)
    factors = tensorlib.divide(nominal_sq, uncerts_sq)
    assert pytest.approx(tensorlib.tolist(par_set_syst.factors)) == tensorlib.tolist(
        factors
    )


@pytest.mark.only_numpy
def test_pdf_integration_histosys(backend):
    with open(
        "validation/data/2bin_histosys_example2.json", encoding="utf-8"
    ) as source_file:
        source = json.load(source_file)
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
    pdf = pyhf.Model(spec)

    pars = [None, None]

    pars[pdf.config.par_slice('mu')], pars[pdf.config.par_slice('bkg_norm')] = [
        [0.0],
        [1.0],
    ]
    assert pdf.expected_data(pars, include_auxdata=False).tolist() == [102, 190]

    pars[pdf.config.par_slice('mu')], pars[pdf.config.par_slice('bkg_norm')] = [
        [0.0],
        [2.0],
    ]
    assert pdf.expected_data(pars, include_auxdata=False).tolist() == [104, 230]

    pars[pdf.config.par_slice('mu')], pars[pdf.config.par_slice('bkg_norm')] = [
        [0.0],
        [-1.0],
    ]
    assert pdf.expected_data(pars, include_auxdata=False).tolist() == [98, 100]

    pars[pdf.config.par_slice('mu')], pars[pdf.config.par_slice('bkg_norm')] = [
        [0.0],
        [-2.0],
    ]
    assert pdf.expected_data(pars, include_auxdata=False).tolist() == [96, 50]

    pars[pdf.config.par_slice('mu')], pars[pdf.config.par_slice('bkg_norm')] = [
        [1.0],
        [1.0],
    ]
    assert pdf.expected_data(pars, include_auxdata=False).tolist() == [
        102 + 30,
        190 + 95,
    ]

    pars[pdf.config.par_slice('mu')], pars[pdf.config.par_slice('bkg_norm')] = [
        [1.0],
        [-1.0],
    ]
    assert pdf.expected_data(pars, include_auxdata=False).tolist() == [
        98 + 30,
        100 + 95,
    ]


def test_pdf_integration_normsys(backend):
    with open(
        "validation/data/2bin_histosys_example2.json", encoding="utf-8"
    ) as source_file:
        source = json.load(source_file)
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
                                'data': {'lo': 0.9, 'hi': 1.1},
                            }
                        ],
                    },
                ],
            }
        ]
    }
    pdf = pyhf.Model(spec)

    pars = [None, None]
    pars[pdf.config.par_slice('mu')], pars[pdf.config.par_slice('bkg_norm')] = [
        [0.0],
        [0.0],
    ]
    assert np.allclose(
        pyhf.tensorlib.tolist(pdf.expected_data(pars, include_auxdata=False)),
        [100, 150],
    )

    pars[pdf.config.par_slice('mu')], pars[pdf.config.par_slice('bkg_norm')] = [
        [0.0],
        [1.0],
    ]
    assert np.allclose(
        pyhf.tensorlib.tolist(pdf.expected_data(pars, include_auxdata=False)),
        [100 * 1.1, 150 * 1.1],
    )

    pars[pdf.config.par_slice('mu')], pars[pdf.config.par_slice('bkg_norm')] = [
        [0.0],
        [-1.0],
    ]
    assert np.allclose(
        pyhf.tensorlib.tolist(pdf.expected_data(pars, include_auxdata=False)),
        [100 * 0.9, 150 * 0.9],
    )


@pytest.mark.only_numpy
def test_pdf_integration_shapesys(backend):
    with open(
        "validation/data/2bin_histosys_example2.json", encoding="utf-8"
    ) as source_file:
        source = json.load(source_file)
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
                            {'name': 'bkg_norm', 'type': 'shapesys', 'data': [10, 10]}
                        ],
                    },
                ],
            }
        ]
    }
    pdf = pyhf.Model(spec)

    pars = [None, None]

    pars[pdf.config.par_slice('mu')], pars[pdf.config.par_slice('bkg_norm')] = [
        [0.0],
        [1.0, 1.0],
    ]
    assert pdf.expected_data(pars, include_auxdata=False).tolist() == [100, 150]

    pars[pdf.config.par_slice('mu')], pars[pdf.config.par_slice('bkg_norm')] = [
        [0.0],
        [1.1, 1.0],
    ]
    assert pdf.expected_data(pars, include_auxdata=False).tolist() == [100 * 1.1, 150]

    pars[pdf.config.par_slice('mu')], pars[pdf.config.par_slice('bkg_norm')] = [
        [0.0],
        [1.0, 1.1],
    ]
    assert pdf.expected_data(pars, include_auxdata=False).tolist() == [100, 150 * 1.1]

    pars[pdf.config.par_slice('mu')], pars[pdf.config.par_slice('bkg_norm')] = [
        [0.0],
        [1.1, 0.9],
    ]
    assert pdf.expected_data(pars, include_auxdata=False).tolist() == [
        100 * 1.1,
        150 * 0.9,
    ]

    pars[pdf.config.par_slice('mu')], pars[pdf.config.par_slice('bkg_norm')] = [
        [0.0],
        [0.9, 1.1],
    ]
    assert pdf.expected_data(pars, include_auxdata=False).tolist() == [
        100 * 0.9,
        150 * 1.1,
    ]


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
                            {
                                'name': 'a_name',
                                'type': 'this_should_not_exist',
                                'data': [1],
                            }
                        ],
                    }
                ],
            }
        ]
    }
    with pytest.raises(pyhf.exceptions.InvalidModifier):
        pyhf.pdf.Model(spec, validate=False)  # don't validate to delay exception


def test_invalid_modifier_name_resuse():
    spec = {
        'channels': [
            {
                'name': 'singlechannel',
                'samples': [
                    {
                        'name': 'signal',
                        'data': [5.0],
                        'modifiers': [
                            {'name': 'reused_name', 'type': 'normfactor', 'data': None}
                        ],
                    },
                    {
                        'name': 'background',
                        'data': [50.0],
                        'modifiers': [
                            {
                                'name': 'reused_name',
                                'type': 'normsys',
                                'data': {'lo': 0.9, 'hi': 1.1},
                            }
                        ],
                    },
                ],
            }
        ]
    }
    with pytest.raises(pyhf.exceptions.InvalidNameReuse):
        pyhf.Model(spec, poi_name='reused_name')


def test_override_paramset_defaults():
    with open(
        "validation/data/2bin_histosys_example2.json", encoding="utf-8"
    ) as source_file:
        source = json.load(source_file)
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
                            {'name': 'bkg_norm', 'type': 'shapesys', 'data': [10, 10]}
                        ],
                    },
                ],
            }
        ],
        'parameters': [
            {'name': 'bkg_norm', 'inits': [99, 99], 'bounds': [[95, 95], [95, 95]]}
        ],
    }
    pdf = pyhf.Model(spec)
    assert pdf.config.param_set('bkg_norm').suggested_bounds == [[95, 95], [95, 95]]
    assert pdf.config.param_set('bkg_norm').suggested_init == [99, 99]


def test_override_paramsets_incorrect_num_parameters():
    with open(
        "validation/data/2bin_histosys_example2.json", encoding="utf-8"
    ) as source_file:
        source = json.load(source_file)
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
                            {'name': 'bkg_norm', 'type': 'shapesys', 'data': [10, 10]}
                        ],
                    },
                ],
            }
        ],
        'parameters': [{'name': 'bkg_norm', 'inits': [99, 99], 'bounds': [[95, 95]]}],
    }
    with pytest.raises(pyhf.exceptions.InvalidModel):
        pyhf.Model(spec)


def test_lumi_np_scaling():
    spec = {
        "channels": [
            {
                "name": "channel1",
                "samples": [
                    {
                        "data": [20.0, 10.0],
                        "modifiers": [
                            {
                                "data": None,
                                "name": "SigXsecOverSM",
                                "type": "normfactor",
                            },
                            {"data": None, "name": "lumi", "type": "lumi"},
                        ],
                        "name": "signal",
                    },
                    {"data": [100.0, 0.0], "name": "background1", "modifiers": []},
                    {
                        "data": [0.0, 100.0],
                        "modifiers": [{"data": None, "name": "lumi", "type": "lumi"}],
                        "name": "background2",
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
    pdf = pyhf.pdf.Model(spec, poi_name="SigXsecOverSM")

    poi_slice = pdf.config.par_slice('SigXsecOverSM')
    lumi_slice = pdf.config.par_slice('lumi')

    index_bkg1 = pdf.config.samples.index('background1')
    index_bkg2 = pdf.config.samples.index('background2')
    index_sig = pdf.config.samples.index('signal')
    bkg1_slice = slice(index_bkg1, index_bkg1 + 1)
    bkg2_slice = slice(index_bkg2, index_bkg2 + 1)
    sig_slice = slice(index_sig, index_sig + 1)

    alpha_lumi = np.random.uniform(0.0, 10.0, 1)[0]

    mods = [None, None, None]
    pars = [None, None]

    pars[poi_slice], pars[lumi_slice] = [[1.0], [1.0]]
    mods[sig_slice], mods[bkg1_slice], mods[bkg2_slice] = [
        [[[1.0, 1.0]]],
        [[[1.0, 1.0]]],
        [[[1.0, 1.0]]],
    ]
    assert pdf._modifications(np.array(pars))[1][0].tolist() == [mods]
    assert pdf.expected_data(pars).tolist() == [120.0, 110.0, 1.0]

    pars[poi_slice], pars[lumi_slice] = [[1.0], [alpha_lumi]]
    mods[sig_slice], mods[bkg1_slice], mods[bkg2_slice] = [
        [[[alpha_lumi, alpha_lumi]]],
        [[[1.0, 1.0]]],
        [[[alpha_lumi, alpha_lumi]]],
    ]
    assert pdf._modifications(np.array(pars))[1][0].tolist() == [mods]
    assert pytest.approx(pdf.expected_data(pars).tolist()) == [
        100 + 20.0 * alpha_lumi,
        110.0 * alpha_lumi,
        1.0 * alpha_lumi,
    ]


def test_sample_wrong_bins():
    spec = {
        'channels': [
            {
                'name': 'channel',
                'samples': [
                    {'name': 'goodsample', 'data': [1.0, 2.0], 'modifiers': []},
                    {'name': 'badsample', 'data': [3.0, 4.0, 5.0], 'modifiers': []},
                ],
            }
        ]
    }
    with pytest.raises(pyhf.exceptions.InvalidModel):
        pyhf.Model(spec)


@pytest.mark.parametrize(
    'measurements, msettings',
    [
        (
            None,
            {'normsys': {'interpcode': 'code4'}, 'histosys': {'interpcode': 'code4p'}},
        )
    ],
)
def test_unexpected_keyword_argument(measurements, msettings):
    spec = {
        "channels": [
            {
                "name": "singlechannel",
                "samples": [
                    {
                        "name": "signal",
                        "data": [5.0, 10.0],
                        "modifiers": [
                            {"name": "mu", "type": "normfactor", "data": None}
                        ],
                    },
                    {
                        "name": "background",
                        "data": [50.0, 60.0],
                        "modifiers": [
                            {
                                "name": "uncorr_bkguncrt",
                                "type": "shapesys",
                                "data": [5.0, 12.0],
                            }
                        ],
                    },
                ],
            }
        ]
    }
    with pytest.raises(pyhf.exceptions.Unsupported):
        pyhf.pdf._ModelConfig(
            spec, measurement_name=measurements, modifiers_settings=msettings
        )


def test_model_integration_fixed_parameters():
    spec = {
        'channels': [
            {
                'name': 'channel',
                'samples': [
                    {
                        'name': 'sample',
                        'data': [10.0],
                        'modifiers': [
                            {'name': 'unfixed', 'type': 'normfactor', 'data': None}
                        ],
                    },
                    {
                        'name': 'another_sample',
                        'data': [5.0],
                        'modifiers': [
                            {'name': 'mypoi', 'type': 'normfactor', 'data': None}
                        ],
                    },
                ],
            }
        ],
        'parameters': [{'name': 'mypoi', 'inits': [1], 'fixed': True}],
    }
    model = pyhf.Model(spec, poi_name='mypoi')
    assert model.config.suggested_fixed()[model.config.par_slice('mypoi')] == [True]


def test_model_integration_fixed_parameters_shapesys():
    spec = {
        'channels': [
            {
                'name': 'channel',
                'samples': [
                    {
                        'name': 'sample',
                        'data': [10.0] * 3,
                        'modifiers': [
                            {'name': 'unfixed', 'type': 'normfactor', 'data': None},
                            {'name': 'uncorr', 'type': 'shapesys', 'data': [1.5] * 3},
                        ],
                    },
                    {
                        'name': 'another_sample',
                        'data': [5.0] * 3,
                        'modifiers': [
                            {'name': 'mypoi', 'type': 'normfactor', 'data': None}
                        ],
                    },
                ],
            }
        ],
        'parameters': [{'name': 'uncorr', 'inits': [1.0, 2.0, 3.0], 'fixed': True}],
    }
    model = pyhf.Model(spec, poi_name='mypoi')
    assert model.config.suggested_fixed()[model.config.par_slice('uncorr')] == [
        True,
        True,
        True,
    ]


def test_reproducible_model_spec():
    ws = {
        "channels": [
            {
                "name": "SR",
                "samples": [
                    {
                        "data": [
                            10.0,
                        ],
                        "modifiers": [
                            {"data": None, "name": "mu", "type": "normfactor"},
                        ],
                        "name": "Signal",
                    }
                ],
            }
        ],
        "measurements": [
            {
                "config": {
                    "parameters": [{"bounds": [[0, 5]], "inits": [1], "name": "mu"}],
                    "poi": "mu",
                },
                "name": "minimal_example",
            }
        ],
        "observations": [{"data": [12], "name": "SR"}],
        "version": "1.0.0",
    }
    workspace = pyhf.Workspace(ws)
    model_from_ws = workspace.model()

    assert model_from_ws.spec['parameters'] == [
        {'bounds': [[0, 5]], 'inits': [1], 'name': 'mu'}
    ]
    assert pyhf.Model(model_from_ws.spec)


def test_par_names_scalar_nonscalar():
    """
    Testing to ensure that nonscalar parameters are still indexed, even if
    n_parameters==1.
    """
    spec = {
        'channels': [
            {
                'name': 'channel',
                'samples': [
                    {
                        'name': 'goodsample',
                        'data': [1.0],
                        'modifiers': [
                            {'type': 'normfactor', 'name': 'scalar', 'data': None},
                            {'type': 'shapesys', 'name': 'nonscalar', 'data': [1.0]},
                        ],
                    },
                ],
            }
        ]
    }

    model = pyhf.Model(spec, poi_name="scalar")
    assert model.config.par_order == ["scalar", "nonscalar"]
    assert model.config.par_names == [
        'scalar',
        'nonscalar[0]',
    ]


def test_make_model_with_tensors():
    def make_model(
        nominal,
        lumi_sigma,
        corrup_data,
        corrdn_data,
        stater_data,
        normsys_up,
        normsys_dn,
        uncorr_data,
    ):
        spec = {
            "channels": [
                {
                    "name": "achannel",
                    "samples": [
                        {
                            "name": "background",
                            "data": nominal,
                            "modifiers": [
                                {"name": "mu", "type": "normfactor", "data": None},
                                {"name": "lumi", "type": "lumi", "data": None},
                                {
                                    "name": "mod_name",
                                    "type": "shapefactor",
                                    "data": None,
                                },
                                {
                                    "name": "corr_bkguncrt2",
                                    "type": "histosys",
                                    "data": {
                                        'hi_data': corrup_data,
                                        'lo_data': corrdn_data,
                                    },
                                },
                                {
                                    "name": "staterror2",
                                    "type": "staterror",
                                    "data": stater_data,
                                },
                                {
                                    "name": "norm",
                                    "type": "normsys",
                                    "data": {'hi': normsys_up, 'lo': normsys_dn},
                                },
                            ],
                        }
                    ],
                },
                {
                    "name": "secondchannel",
                    "samples": [
                        {
                            "name": "background",
                            "data": nominal,
                            "modifiers": [
                                {"name": "mu", "type": "normfactor", "data": None},
                                {"name": "lumi", "type": "lumi", "data": None},
                                {
                                    "name": "mod_name",
                                    "type": "shapefactor",
                                    "data": None,
                                },
                                {
                                    "name": "uncorr_bkguncrt2",
                                    "type": "shapesys",
                                    "data": uncorr_data,
                                },
                                {
                                    "name": "corr_bkguncrt2",
                                    "type": "histosys",
                                    "data": {
                                        'hi_data': corrup_data,
                                        'lo_data': corrdn_data,
                                    },
                                },
                                {
                                    "name": "staterror",
                                    "type": "staterror",
                                    "data": stater_data,
                                },
                                {
                                    "name": "norm",
                                    "type": "normsys",
                                    "data": {'hi': normsys_up, 'lo': normsys_dn},
                                },
                            ],
                        }
                    ],
                },
            ],
        }
        model = pyhf.Model(
            {
                'channels': spec['channels'],
                'parameters': [
                    {
                        'name': 'lumi',
                        'auxdata': [1.0],
                        'bounds': [[0.5, 1.5]],
                        'inits': [1.0],
                        "sigmas": [lumi_sigma],
                    }
                ],
            },
            validate=False,
        )

        pars = model.config.suggested_init()
        exp_data = model.expected_data(pars)
        assert exp_data is not None

    make_model(
        pyhf.tensorlib.astensor([60.0, 62.0]),
        pyhf.tensorlib.astensor(0.2),
        pyhf.tensorlib.astensor([60.0, 62.0]),
        pyhf.tensorlib.astensor([60.0, 62.0]),
        pyhf.tensorlib.astensor([5.0, 5.0]),
        pyhf.tensorlib.astensor(0.95),
        pyhf.tensorlib.astensor(1.05),
        pyhf.tensorlib.astensor([5.0, 5.0]),
    )


def test_pdf_clipping(backend):
    tensorlib, optimizer = pyhf.get_backend()

    spec = {
        "channels": [
            {
                "name": "ch1",
                "samples": [
                    {
                        "data": [100.0, 100.0],
                        "modifiers": [
                            {"data": None, "name": "mu_sig", "type": "normfactor"},
                            {
                                "data": {
                                    "hi_data": [125.0, 75.0],
                                    "lo_data": [175.0, 10.0],
                                },
                                "name": "np_1",
                                "type": "histosys",
                            },
                            {
                                "data": {
                                    "hi_data": [125.0, 75.0],
                                    "lo_data": [175.0, 10.0],
                                },
                                "name": "np_2",
                                "type": "histosys",
                            },
                        ],
                        "name": "signal",
                    }
                ],
            },
            {
                "name": "ch2",
                "samples": [
                    {
                        "data": [15000.0],
                        "modifiers": [
                            {"data": None, "name": "mu_sig", "type": "normfactor"},
                            {
                                "data": {"hi_data": [15500.0], "lo_data": [1200.0]},
                                "name": "np_1",
                                "type": "histosys",
                            },
                            {
                                "data": {"hi_data": [15500.0], "lo_data": [1200.0]},
                                "name": "np_2",
                                "type": "histosys",
                            },
                        ],
                        "name": "signal",
                    }
                ],
            },
        ],
        "measurements": [
            {"config": {"parameters": [], "poi": "mu_sig"}, "name": "meas"}
        ],
        "observations": [
            {"data": [100], "name": "ch1"},
            {"data": [1000], "name": "ch2"},
        ],
        "version": "1.0.0",
    }

    par_values = []

    ws = pyhf.Workspace(spec)
    model = ws.model()
    data = tensorlib.astensor([100.0, 100.0, 10.0, 0.0, 0.0])

    for par_name in model.config.par_names:
        if "np" in par_name:
            par_values.append(-0.6)  # np_1 / np_2
        else:
            par_values.append(1.0)  # mu

    pars = tensorlib.astensor(par_values)

    # Check with no clipping
    assert any(value < 0 for value in model.expected_actualdata(pars))
    assert tensorlib.tolist(model.expected_actualdata(pars)) == pytest.approx(
        [1.830496e02, -7.040000e-03, -6.355968e02], abs=1e-3
    )

    # Check with clipping by-sample
    model_clip_sample = ws.model(clip_sample_data=0.0)
    assert all(value >= 0 for value in model_clip_sample.expected_actualdata(pars))
    assert tensorlib.tolist(
        model_clip_sample.expected_actualdata(pars)
    ) == pytest.approx([1.830496e02, 0.0, 0.0], abs=1e-3)

    # Check with clipping by-bin
    model_clip_bin = ws.model(clip_bin_data=0.0)
    assert all(value >= 0 for value in model_clip_bin.expected_actualdata(pars))
    assert tensorlib.tolist(model_clip_bin.expected_actualdata(pars)) == pytest.approx(
        [1.830496e02, 0.0, 0.0], abs=1e-3
    )

    # Minuit cannot handle negative yields, confirm that MLE fails for minuit specifically
    if optimizer.name == 'minuit':
        with pytest.raises(pyhf.exceptions.FailedMinimization):
            pyhf.infer.mle.fit(data, model)
    else:
        pyhf.infer.mle.fit(data, model)

    # We should be able to converge when clipping is enabled
    pyhf.infer.mle.fit(data, model_clip_sample)
    pyhf.infer.mle.fit(data, model_clip_bin)


def test_is_shared_paramset_shapesys_diff_sample_diff_channel():
    spec = {
        "channels": [
            {
                "name": "SR",
                "samples": [
                    {
                        "data": [24.0, 25.0],
                        "modifiers": [
                            {"data": [0.1, 0.2], "name": "par", "type": "shapesys"},
                            {"data": None, "name": "mu", "type": "normfactor"},
                        ],
                        "name": "Signal",
                    }
                ],
            },
            {
                "name": "CR",
                "samples": [
                    {
                        "data": [10.0],
                        "modifiers": [
                            {"data": [0.1], "name": "par", "type": "shapesys"}
                        ],
                        "name": "Background",
                    }
                ],
            },
        ],
        "measurements": [
            {"config": {"parameters": [], "poi": "mu"}, "name": "minimal_example"}
        ],
        "observations": [
            {"data": [24.0, 24.0], "name": "SR"},
            {"data": [10.0], "name": "CR"},
        ],
        "version": "1.0.0",
    }

    with pytest.raises(pyhf.exceptions.InvalidModel):
        pyhf.Workspace(spec).model()


def test_is_shared_paramset_shapesys_diff_sample_same_channel():
    spec = {
        "channels": [
            {
                "name": "SR",
                "samples": [
                    {
                        "data": [50],
                        "modifiers": [
                            {
                                "data": [9],
                                "name": "abc",
                                "type": "shapesys",
                            },
                            {
                                "data": None,
                                "name": "Signal strength",
                                "type": "normfactor",
                            },
                        ],
                        "name": "Signal",
                    },
                    {
                        "data": [150],
                        "modifiers": [
                            {
                                "data": [7],
                                "name": "abc",
                                "type": "shapesys",
                            }
                        ],
                        "name": "Background",
                    },
                ],
            }
        ],
        "measurements": [{"config": {"parameters": [], "poi": ""}, "name": "meas"}],
        "observations": [{"data": [160], "name": "SR"}],
        "version": "1.0.0",
    }

    with pytest.raises(pyhf.exceptions.InvalidModel):
        pyhf.Workspace(spec).model()


def test_is_shared_paramset_shapesys_same_sample_same_channel():
    spec = {
        "channels": [
            {
                "name": "SR",
                "samples": [
                    {
                        "data": [24.0, 25.0],
                        "modifiers": [
                            {"data": [0.1, 0.2], "name": "par2", "type": "shapesys"},
                            {"data": None, "name": "mu", "type": "normfactor"},
                        ],
                        "name": "Signal",
                    }
                ],
            },
            {
                "name": "CR",
                "samples": [
                    {
                        "data": [10.0],
                        "modifiers": [
                            {"data": [0.1], "name": "par", "type": "shapesys"},
                            {"data": [0.5], "name": "par", "type": "shapesys"},
                        ],
                        "name": "Background",
                    }
                ],
            },
        ],
        "measurements": [
            {"config": {"parameters": [], "poi": "mu"}, "name": "minimal_example"}
        ],
        "observations": [
            {"data": [24.0, 24.0], "name": "SR"},
            {"data": [10.0], "name": "CR"},
        ],
        "version": "1.0.0",
    }

    with pytest.raises(pyhf.exceptions.InvalidModel):
        pyhf.Workspace(spec).model()

import pyhf
import pytest
import pyhf.exceptions
import numpy as np
import json


def test_pdf_inputs(backend):
    source = {
        "binning": [2, -0.5, 1.5],
        "bindata": {"data": [55.0], "bkg": [50.0], "bkgerr": [7.0], "sig": [10.0]},
    }
    pdf = pyhf.simplemodels.hepdata_like(
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
    pdf = pyhf.simplemodels.hepdata_like(
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
    pdf = pyhf.simplemodels.hepdata_like(
        source['bindata']['sig'], source['bindata']['bkg'], source['bindata']['bkgerr']
    )

    pars = pdf.config.suggested_init()
    data = source['bindata']['data'] + [10.0] + pdf.config.auxdata

    with pytest.raises(pyhf.exceptions.InvalidPdfData):
        pdf.logpdf(pars, data)


def test_pdf_basicapi_tests(backend):
    source = {
        "binning": [2, -0.5, 1.5],
        "bindata": {"data": [55.0], "bkg": [50.0], "bkgerr": [7.0], "sig": [10.0]},
    }
    pdf = pyhf.simplemodels.hepdata_like(
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

    pdf = pyhf.simplemodels.hepdata_like(
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
                            },
                            {
                                "data": [0, 0, 0, 0, 0, 0],
                                "name": "syst_lowstats",
                                "type": "shapesys",
                            },
                        ],
                        "name": "background1",
                    },
                ],
            }
        ]
    }
    pdf = pyhf.Model(spec)
    par_set_syst = pdf.config.param_set('syst')
    par_set_syst_lowstats = pdf.config.param_set('syst_lowstats')

    assert par_set_syst.n_parameters == 4
    assert par_set_syst_lowstats.n_parameters == 0
    tensorlib, _ = backend
    nominal_sq = tensorlib.power(tensorlib.astensor([100.0, 90, 0.0, 70, 0.1, 50]), 2)
    uncerts_sq = tensorlib.power(tensorlib.astensor([10, 9, 1, 0.0, 0.1, 5]), 2)
    factors = tensorlib.divide(nominal_sq, uncerts_sq)
    indices = tensorlib.astensor([0, 1, 4, 5], dtype='int')
    assert pytest.approx(tensorlib.tolist(par_set_syst.factors)) == tensorlib.tolist(
        tensorlib.gather(factors, indices)
    )
    assert getattr(par_set_syst_lowstats, 'factors', None) is None


@pytest.mark.only_numpy
def test_pdf_integration_histosys(backend):
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
        pyhf.pdf._ModelConfig(spec)


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
        pyhf.Model(spec, poiname='reused_name')


def test_override_paramset_defaults():
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
    pdf = pyhf.pdf.Model(spec, poiname="SigXsecOverSM")

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

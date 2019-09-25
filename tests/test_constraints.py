import pytest
import pyhf
from pyhf.parameters import constrained_by_poisson, constrained_by_normal
from pyhf.constraints import gaussian_constraint_combined, poisson_constraint_combined
from pyhf import default_backend
import numpy as np


class MockConfig(object):
    def __init__(self, par_map, par_order):
        self.par_order = par_order
        self.par_map = par_map

        self.auxdata = []
        self.auxdata_order = []
        for name in self.par_order:
            self.auxdata = self.auxdata + self.par_map[name]['paramset'].auxdata
            self.auxdata_order.append(name)
        self.npars = len(self.suggested_init())

    def suggested_init(self):
        init = []
        for name in self.par_order:
            init = init + self.par_map[name]['paramset'].suggested_init
        return init

    def par_slice(self, name):
        return self.par_map[name]['slice']

    def param_set(self, name):
        return self.par_map[name]['paramset']


def test_numpy_pdf_inputs(backend):
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
                    {
                        'name': 'bkg3',
                        'data': [20.0, 15.0],
                        'modifiers': [
                            {'name': 'bkg_norm', 'type': 'shapesys', 'data': [10, 10]}
                        ],
                    },
                ],
            }
        ]
    }

    m = pyhf.Model(spec)

    def slow(self, auxdata, pars):
        tensorlib, _ = pyhf.get_backend()
        # iterate over all constraints order doesn't matter....
        start_index = 0
        summands = None
        for cname in self.config.auxdata_order:
            parset, parslice = (
                self.config.param_set(cname),
                self.config.par_slice(cname),
            )
            end_index = start_index + parset.n_parameters
            thisauxdata = auxdata[start_index:end_index]
            start_index = end_index
            if parset.pdf_type == 'normal':
                paralphas = pars[parslice]
                sigmas = (
                    parset.sigmas
                    if hasattr(parset, 'sigmas')
                    else tensorlib.ones(paralphas.shape)
                )
                sigmas = tensorlib.astensor(sigmas)
                constraint_term = tensorlib.normal_logpdf(
                    thisauxdata, paralphas, sigmas
                )
            elif parset.pdf_type == 'poisson':
                paralphas = tensorlib.product(
                    tensorlib.stack(
                        [pars[parslice], tensorlib.astensor(parset.factors)]
                    ),
                    axis=0,
                )

                constraint_term = tensorlib.poisson_logpdf(thisauxdata, paralphas)
            summands = (
                constraint_term
                if summands is None
                else tensorlib.concatenate([summands, constraint_term])
            )
        return tensorlib.sum(summands) if summands is not None else 0

    def fast(self, auxdata, pars):
        return self.constraint_logpdf(auxdata, pars)

    auxd = pyhf.tensorlib.astensor(m.config.auxdata)
    pars = pyhf.tensorlib.astensor(m.config.suggested_init())
    slow_result = pyhf.tensorlib.tolist(slow(m, auxd, pars))
    fast_result = pyhf.tensorlib.tolist(fast(m, auxd, pars))
    assert pytest.approx(slow_result) == fast_result


def test_batched_constraints(backend):
    config = MockConfig(
        par_order=['pois1', 'pois2', 'norm1', 'norm2'],
        par_map={
            'pois1': {
                'paramset': constrained_by_poisson(
                    n_parameters=1,
                    inits=[1.0],
                    bounds=[[0, 10]],
                    auxdata=[12],
                    factors=[12],
                ),
                'slice': slice(0, 1),
                'auxdata': [1],
            },
            'pois2': {
                'paramset': constrained_by_poisson(
                    n_parameters=2,
                    inits=[1.0] * 2,
                    bounds=[[0, 10]] * 2,
                    auxdata=[13, 14],
                    factors=[13, 14],
                ),
                'slice': slice(1, 3),
            },
            'norm1': {
                'paramset': constrained_by_normal(
                    n_parameters=2,
                    inits=[0] * 2,
                    bounds=[[0, 10]] * 2,
                    auxdata=[0, 0],
                    sigmas=[1.5, 2.0],
                ),
                'slice': slice(3, 5),
            },
            'norm2': {
                'paramset': constrained_by_normal(
                    n_parameters=3,
                    inits=[0] * 3,
                    bounds=[[0, 10]] * 3,
                    auxdata=[0, 0, 0],
                ),
                'slice': slice(5, 8),
            },
        },
    )
    suggested_pars = [1.0] * 3 + [0.0] * 5  # 2 pois 5 norm
    constraint = poisson_constraint_combined(config)
    result = default_backend.astensor(
        pyhf.tensorlib.tolist(
            constraint.logpdf(
                pyhf.tensorlib.astensor(config.auxdata),
                pyhf.tensorlib.astensor(suggested_pars),
            )
        )
    )
    assert np.isclose(
        result[0],
        sum(
            [
                default_backend.poisson_logpdf(data, rate)
                for data, rate in zip([12, 13, 14], [12, 13, 14])
            ]
        ),
    )
    assert result.shape == (1,)

    suggested_pars = [1.1] * 3 + [0.0] * 5  # 2 pois 5 norm
    constraint = poisson_constraint_combined(config)
    result = default_backend.astensor(
        pyhf.tensorlib.tolist(
            constraint.logpdf(
                pyhf.tensorlib.astensor(config.auxdata),
                pyhf.tensorlib.astensor(suggested_pars),
            )
        )
    )
    assert np.isclose(
        result[0],
        sum(
            [
                default_backend.poisson_logpdf(data, rate)
                for data, rate in zip([12, 13, 14], [12 * 1.1, 13 * 1.1, 14 * 1.1])
            ]
        ),
    )
    assert result.shape == (1,)

    constraint = poisson_constraint_combined(config, batch_size=10)
    result = constraint.logpdf(
        pyhf.tensorlib.astensor(config.auxdata),
        pyhf.tensorlib.astensor([suggested_pars] * 10),
    )
    assert result.shape == (10,)

    suggested_pars = [
        [1.1, 1.2, 1.3] + [0.0] * 5,  # 2 pois 5 norm
        [0.7, 0.8, 0.9] + [0.0] * 5,  # 2 pois 5 norm
        [0.4, 0.5, 0.6] + [0.0] * 5,  # 2 pois 5 norm
    ]
    constraint = poisson_constraint_combined(config, batch_size=3)
    result = default_backend.astensor(
        pyhf.tensorlib.tolist(
            constraint.logpdf(
                pyhf.tensorlib.astensor(config.auxdata),
                pyhf.tensorlib.astensor(suggested_pars),
            )
        )
    )
    assert np.all(
        np.isclose(
            result,
            np.sum(
                [
                    [
                        default_backend.poisson_logpdf(data, rate)
                        for data, rate in zip(
                            [12, 13, 14], [12 * 1.1, 13 * 1.2, 14 * 1.3]
                        )
                    ],
                    [
                        default_backend.poisson_logpdf(data, rate)
                        for data, rate in zip(
                            [12, 13, 14], [12 * 0.7, 13 * 0.8, 14 * 0.9]
                        )
                    ],
                    [
                        default_backend.poisson_logpdf(data, rate)
                        for data, rate in zip(
                            [12, 13, 14], [12 * 0.4, 13 * 0.5, 14 * 0.6]
                        )
                    ],
                ],
                axis=1,
            ),
        )
    )
    assert result.shape == (3,)

    suggested_pars = [1.0] * 3 + [0.0] * 5  # 2 pois 5 norm
    constraint = gaussian_constraint_combined(config, batch_size=1)
    result = default_backend.astensor(
        pyhf.tensorlib.tolist(
            constraint.logpdf(
                pyhf.tensorlib.astensor(config.auxdata),
                pyhf.tensorlib.astensor(suggested_pars),
            )
        )
    )
    assert np.isclose(
        result[0],
        sum(
            [
                default_backend.normal_logpdf(data, mu, sigma)
                for data, mu, sigma in zip(
                    [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1.5, 2.0, 1.0, 1.0, 1.0]
                )
            ]
        ),
    )
    assert result.shape == (1,)

    suggested_pars = [1.0] * 3 + [1, 2, 3, 4, 5]  # 2 pois 5 norm
    constraint = gaussian_constraint_combined(config, batch_size=1)
    result = default_backend.astensor(
        pyhf.tensorlib.tolist(
            constraint.logpdf(
                pyhf.tensorlib.astensor(config.auxdata),
                pyhf.tensorlib.astensor(suggested_pars),
            )
        )
    )
    assert np.isclose(
        result[0],
        sum(
            [
                default_backend.normal_logpdf(data, mu, sigma)
                for data, mu, sigma in zip(
                    [0, 0, 0, 0, 0], [1, 2, 3, 4, 5], [1.5, 2.0, 1.0, 1.0, 1.0]
                )
            ]
        ),
    )
    assert result.shape == (1,)

    suggested_pars = [
        [1.0] * 3 + [1, 2, 3, 4, 5],  # 2 pois 5 norm
        [1.0] * 3 + [-1, -2, -3, -4, -5],  # 2 pois 5 norm
        [1.0] * 3 + [-1, -2, 0, 1, 2],  # 2 pois 5 norm
    ]
    constraint = gaussian_constraint_combined(config, batch_size=3)
    result = default_backend.astensor(
        pyhf.tensorlib.tolist(
            constraint.logpdf(
                pyhf.tensorlib.astensor(config.auxdata),
                pyhf.tensorlib.astensor(suggested_pars),
            )
        )
    )
    assert np.all(
        np.isclose(
            result,
            np.sum(
                [
                    [
                        default_backend.normal_logpdf(data, mu, sigma)
                        for data, mu, sigma in zip(
                            [0, 0, 0, 0, 0], [1, 2, 3, 4, 5], [1.5, 2.0, 1.0, 1.0, 1.0]
                        )
                    ],
                    [
                        default_backend.normal_logpdf(data, mu, sigma)
                        for data, mu, sigma in zip(
                            [0, 0, 0, 0, 0],
                            [-1, -2, -3, -4, -5],
                            [1.5, 2.0, 1.0, 1.0, 1.0],
                        )
                    ],
                    [
                        default_backend.normal_logpdf(data, mu, sigma)
                        for data, mu, sigma in zip(
                            [0, 0, 0, 0, 0],
                            [-1, -2, 0, 1, 2],
                            [1.5, 2.0, 1.0, 1.0, 1.0],
                        )
                    ],
                ],
                axis=1,
            ),
        )
    )
    assert result.shape == (3,)

    constraint = gaussian_constraint_combined(config, batch_size=10)
    result = constraint.logpdf(
        pyhf.tensorlib.astensor(config.auxdata),
        pyhf.tensorlib.astensor([suggested_pars] * 10),
    )
    assert result.shape == (10,)

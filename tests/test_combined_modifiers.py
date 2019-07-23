from pyhf.pdf import _ModelConfig
from pyhf.modifiers.histosys import histosys_combined
from pyhf.modifiers.normsys import normsys_combined
from pyhf.modifiers.lumi import lumi_combined
from pyhf.modifiers.staterror import staterror_combined
from pyhf.modifiers.shapesys import shapesys_combined
from pyhf.modifiers.normfactor import normfactor_combined
from pyhf.modifiers.shapefactor import shapefactor_combined
from pyhf.paramsets import paramset
import numpy as np
import pyhf
import pytest


class MockConfig(object):
    def __init__(self, par_map, par_order, samples, channels=None, channel_nbins=None):
        self.par_order = par_order
        self.par_map = par_map
        self.samples = samples
        self.channels = channels
        self.channel_nbins = channel_nbins

    def suggested_init(self):
        init = []
        for name in self.par_order:
            init = init + self.par_map[name]['paramset'].suggested_init
        return init

    def suggested_bounds(self):
        bounds = []
        for name in self.par_order:
            bounds = bounds + self.par_map[name]['paramset'].suggested_bounds
        return bounds

    def par_slice(self, name):
        return self.par_map[name]['slice']

    def param_set(self, name):
        return self.par_map[name]['paramset']


@pytest.mark.skip_mxnet
def test_histosys(backend):
    mc = MockConfig(
        par_map={
            'hello': {
                'paramset': paramset(n_parameters=1, inits=[0], bounds=[[-5, 5]]),
                'slice': slice(0, 1),
            },
            'world': {
                'paramset': paramset(n_parameters=1, inits=[0], bounds=[[-5, 5]]),
                'slice': slice(1, 2),
            },
        },
        par_order=['hello', 'world'],
        samples=['signal', 'background'],
    )

    mega_mods = {
        'signal': {
            'histosys/hello': {
                'type': 'histosys',
                'name': 'hello',
                'data': {
                    'hi_data': [11, 12, 13],
                    'lo_data': [9, 8, 7],
                    'nom_data': [10, 10, 10],
                    'mask': [True, True, True],
                },
            },
            'histosys/world': {
                'type': 'histosys',
                'name': 'world',
                'data': {
                    'hi_data': [10, 10, 10],
                    'lo_data': [5, 6, 7],
                    'nom_data': [10, 10, 10],
                    'mask': [True, True, True],
                },
            },
        },
        'background': {
            'histosys/hello': {
                'type': 'histosys',
                'name': 'hello',
                'data': {
                    'hi_data': [11, 12, 13],
                    'lo_data': [9, 8, 7],
                    'nom_data': [10, 10, 10],
                    'mask': [True, True, True],
                },
            },
            'histosys/world': {
                'type': 'histosys',
                'name': 'world',
                'data': {
                    'hi_data': [10, 10, 10],
                    'lo_data': [5, 6, 7],
                    'nom_data': [10, 10, 10],
                    'mask': [True, True, True],
                },
            },
        },
    }

    hsc = histosys_combined(
        [('hello', 'histosys'), ('world', 'histosys')], mc, mega_mods
    )

    mod = hsc.apply(pyhf.tensorlib.astensor([0.5, -1.0]))
    shape = pyhf.tensorlib.shape(mod)
    assert shape == (2, 2, 1, 3)
    mod = np.asarray(pyhf.tensorlib.tolist(mod))
    assert np.allclose(mod[0, 0, 0], [0.5, 1.0, 1.5])


@pytest.mark.skip_mxnet
def test_normsys(backend):
    mc = MockConfig(
        par_map={
            'hello': {
                'paramset': paramset(n_parameters=1, inits=[0], bounds=[[-5, 5]]),
                'slice': slice(0, 1),
            },
            'world': {
                'paramset': paramset(n_parameters=1, inits=[0], bounds=[[-5, 5]]),
                'slice': slice(1, 2),
            },
        },
        par_order=['hello', 'world'],
        samples=['signal', 'background'],
    )

    mega_mods = {
        'signal': {
            'normsys/hello': {
                'type': 'normsys',
                'name': 'hello',
                'data': {
                    'hi': [1.1] * 3,
                    'lo': [0.9] * 3,
                    'nom_data': [1, 1, 1],
                    'mask': [True, True, True],
                },
            },
            'normsys/world': {
                'type': 'v',
                'name': 'world',
                'data': {
                    'hi': [1.3] * 3,
                    'lo': [0.7] * 3,
                    'nom_data': [1, 1, 1],
                    'mask': [True, True, True],
                },
            },
        },
        'background': {
            'normsys/hello': {
                'type': 'normsys',
                'name': 'hello',
                'data': {
                    'hi': [1.2] * 3,
                    'lo': [0.8] * 3,
                    'nom_data': [1, 1, 1],
                    'mask': [True, True, True],
                },
            },
            'normsys/world': {
                'type': 'normsys',
                'name': 'world',
                'data': {
                    'hi': [1.4] * 3,
                    'lo': [0.6] * 3,
                    'nom_data': [1, 1, 1],
                    'mask': [True, True, True],
                },
            },
        },
    }

    hsc = normsys_combined([('hello', 'normsys'), ('world', 'normsys')], mc, mega_mods)

    mod = hsc.apply(pyhf.tensorlib.astensor([1.0, -1.0]))
    shape = pyhf.tensorlib.shape(mod)
    assert shape == (2, 2, 1, 3)
    mod = np.asarray(pyhf.tensorlib.tolist(mod))
    assert np.allclose(mod[0, 0, 0], [1.1, 1.1, 1.1])
    assert np.allclose(mod[0, 1, 0], [1.2, 1.2, 1.2])
    assert np.allclose(mod[1, 0, 0], [0.7, 0.7, 0.7])
    assert np.allclose(mod[1, 1, 0], [0.6, 0.6, 0.6])


@pytest.mark.skip_mxnet
def test_lumi(backend):
    mc = MockConfig(
        par_map={
            'lumi': {
                'paramset': paramset(n_parameters=1, inits=[0], bounds=[[-5, 5]]),
                'slice': slice(0, 1),
            }
        },
        par_order=['lumi'],
        samples=['signal', 'background'],
    )

    mega_mods = {
        'signal': {
            'lumi/lumi': {
                'type': 'lumi',
                'name': 'lumi',
                'data': {'mask': [True, True, True]},
            }
        },
        'background': {
            'lumi/lumi': {
                'type': 'lumi',
                'name': 'lumi',
                'data': {'mask': [True, True, True]},
            }
        },
    }

    hsc = lumi_combined([('lumi', 'lumi')], mc, mega_mods)

    mod = hsc.apply(pyhf.tensorlib.astensor([0.5]))
    shape = pyhf.tensorlib.shape(mod)
    assert shape == (1, 2, 1, 3)

    mod = np.asarray(pyhf.tensorlib.tolist(mod))
    assert np.allclose(mod[0, 0, 0], [0.5, 0.5, 0.5])
    assert np.allclose(mod[0, 1, 0], [0.5, 0.5, 0.5])


@pytest.mark.skip_mxnet
def test_stat(backend):
    mc = MockConfig(
        par_map={
            'staterror_chan1': {
                'paramset': paramset(n_parameters=1, inits=[1], bounds=[[0, 10]]),
                'slice': slice(0, 1),
            },
            'staterror_chan2': {
                'paramset': paramset(
                    n_parameters=2, inits=[1, 1], bounds=[[0, 10], [0, 10]]
                ),
                'slice': slice(1, 3),
            },
        },
        par_order=['staterror_chan1', 'staterror_chan2'],
        samples=['signal', 'background'],
    )

    mega_mods = {
        'signal': {
            'staterror/staterror_chan1': {
                'type': 'staterror',
                'name': 'staterror_chan1',
                'data': {
                    'mask': [True, False, False],
                    'nom_data': [10, 10, 10],
                    'uncrt': [1, 0, 0],
                },
            },
            'staterror/staterror_chan2': {
                'type': 'staterror',
                'name': 'staterror_chan2',
                'data': {
                    'mask': [False, True, True],
                    'nom_data': [10, 10, 10],
                    'uncrt': [0, 1, 1],
                },
            },
        },
        'background': {
            'staterror/staterror_chan1': {
                'type': 'staterror',
                'name': 'staterror_chan1',
                'data': {
                    'mask': [True, False, False],
                    'nom_data': [10, 10, 10],
                    'uncrt': [1, 0, 0],
                },
            },
            'staterror/staterror_chan2': {
                'type': 'staterror',
                'name': 'c',
                'data': {
                    'mask': [False, True, True],
                    'nom_data': [10, 10, 10],
                    'uncrt': [0, 1, 1],
                },
            },
        },
    }
    hsc = staterror_combined(
        [('staterror_chan1', 'staterror'), ('staterror_chan2', 'staterror')],
        mc,
        mega_mods,
    )

    mod = hsc.apply(pyhf.tensorlib.astensor([1.1, 1.2, 1.3]))
    shape = pyhf.tensorlib.shape(mod)
    assert shape == (2, 2, 1, 3)

    mod = np.asarray(pyhf.tensorlib.tolist(mod))
    assert np.allclose(mod[0, 0, 0], [1.1, 1.0, 1.0])
    assert np.allclose(mod[1, 0, 0], [1, 1.2, 1.3])


@pytest.mark.skip_mxnet
def test_shapesys(backend):
    mc = MockConfig(
        par_map={
            'shapesys1': {
                'paramset': paramset(n_parameters=1, inits=[0], bounds=[[0, 10]]),
                'slice': slice(0, 1),
            },
            'shapesys2': {
                'paramset': paramset(
                    n_parameters=2, inits=[0, 0], bounds=[[0, 10], [0, 10]]
                ),
                'slice': slice(1, 3),
            },
        },
        par_order=['shapesys1', 'shapesys2'],
        samples=['signal', 'background'],
    )

    mega_mods = {
        'signal': {
            'shapesys/shapesys1': {
                'type': 'shapesys',
                'name': 'shapesys1',
                'data': {
                    'mask': [True, False, False],
                    'nom_data': [10, 10, 10],
                    'uncrt': [1, 0, 0],
                },
            },
            'shapesys/shapesys2': {
                'type': 'shapesys',
                'name': 'shapesys1',
                'data': {
                    'mask': [False, True, True],
                    'nom_data': [10, 10, 10],
                    'uncrt': [0, 1, 1],
                },
            },
        },
        'background': {
            'shapesys/shapesys1': {
                'type': 'shapesys',
                'name': 'shapesys1',
                'data': {
                    'mask': [True, False, False],
                    'nom_data': [10, 10, 10],
                    'uncrt': [1, 0, 0],
                },
            },
            'shapesys/shapesys2': {
                'type': 'shapesys',
                'name': 'shapesys1',
                'data': {
                    'mask': [False, True, True],
                    'nom_data': [10, 10, 10],
                    'uncrt': [0, 1, 1],
                },
            },
        },
    }
    hsc = staterror_combined(
        [('shapesys1', 'shapesys'), ('shapesys2', 'shapesys')], mc, mega_mods
    )

    mod = hsc.apply(pyhf.tensorlib.astensor([1.1, 1.2, 1.3]))
    shape = pyhf.tensorlib.shape(mod)
    assert shape == (2, 2, 1, 3)

    mod = np.asarray(pyhf.tensorlib.tolist(mod))
    assert np.allclose(mod[0, 0, 0], [1.1, 1.0, 1.0])
    assert np.allclose(mod[1, 0, 0], [1, 1.2, 1.3])


@pytest.mark.skip_mxnet
def test_normfactor(backend):
    mc = MockConfig(
        par_map={
            'mu1': {
                'paramset': paramset(n_parameters=1, inits=[0], bounds=[[0, 10]]),
                'slice': slice(0, 1),
            },
            'mu2': {
                'paramset': paramset(n_parameters=1, inits=[0], bounds=[[0, 10]]),
                'slice': slice(1, 2),
            },
        },
        par_order=['mu1', 'mu2'],
        samples=['signal', 'background'],
    )

    mega_mods = {
        'signal': {
            'normfactor/mu1': {
                'type': 'normfactor',
                'name': 'mu1',
                'data': {'mask': [True, False, False]},
            },
            'normfactor/mu2': {
                'type': 'normfactor',
                'name': 'mu2',
                'data': {'mask': [False, True, True]},
            },
        },
        'background': {
            'normfactor/mu1': {
                'type': 'normfactor',
                'name': 'mu1',
                'data': {'mask': [True, False, False]},
            },
            'normfactor/mu2': {
                'type': 'normfactor',
                'name': 'mu2',
                'data': {'mask': [False, True, True]},
            },
        },
    }
    hsc = normfactor_combined(
        [('mu1', 'normfactor'), ('mu2', 'normfactor')], mc, mega_mods
    )

    mod = hsc.apply(pyhf.tensorlib.astensor([2.0, 3.0]))
    shape = pyhf.tensorlib.shape(mod)
    assert shape == (2, 2, 1, 3)

    mod = np.asarray(pyhf.tensorlib.tolist(mod))
    assert np.allclose(mod[0, 0, 0], [2.0, 1.0, 1.0])
    assert np.allclose(mod[1, 0, 0], [1.0, 3.0, 3.0])


@pytest.mark.skip_mxnet
def test_shapefactor(backend):
    mc = MockConfig(
        par_map={
            'shapefac1': {
                'paramset': paramset(n_parameters=1, inits=[0], bounds=[[0, 10]]),
                'slice': slice(0, 1),
            },
            'shapefac2': {
                'paramset': paramset(
                    n_parameters=2, inits=[0, 0], bounds=[[0, 10], [0, 10]]
                ),
                'slice': slice(1, 3),
            },
        },
        par_order=['shapefac1', 'shapefac2'],
        samples=['signal', 'background'],
        channels=['chan_one', 'chan_two'],
        channel_nbins={'chan_one': 1, 'chan_two': 2},
    )

    mega_mods = {
        'signal': {
            'shapefactor/shapefac1': {
                'type': 'shapefactor',
                'name': 'shapefac1',
                'data': {'mask': [True, False, False]},
            },
            'shapefactor/shapefac2': {
                'type': 'shapefactor',
                'name': 'shapefac2',
                'data': {'mask': [False, True, True]},
            },
        },
        'background': {
            'shapefactor/shapefac1': {
                'type': 'shapefactor',
                'name': 'shapefac1',
                'data': {'mask': [True, False, False]},
            },
            'shapefactor/shapefac2': {
                'type': 'normfactor',
                'name': 'shapefac2',
                'data': {'mask': [False, True, True]},
            },
        },
    }
    hsc = shapefactor_combined(
        [('shapefac1', 'shapefactor'), ('shapefac2', 'shapefactor')], mc, mega_mods
    )

    mod = hsc.apply(pyhf.tensorlib.astensor([2.0, 3.0, 4.0]))
    shape = pyhf.tensorlib.shape(mod)
    assert shape == (2, 2, 1, 3)

    mod = np.asarray(pyhf.tensorlib.tolist(mod))
    assert np.allclose(mod[0, 0, 0], [2.0, 1.0, 1.0])
    assert np.allclose(mod[1, 0, 0], [1.0, 3.0, 4.0])

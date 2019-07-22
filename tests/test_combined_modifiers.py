
from pyhf.pdf import _ModelConfig
from pyhf.modifiers.histosys import histosys_combined
from pyhf.modifiers.normsys import normsys_combined
from pyhf.modifiers.lumi import lumi_combined
from pyhf.paramsets import paramset
import pyhf
import pytest

class MockConfig(object):
    def __init__(self, par_map, par_order, samples):
        self.par_order = par_order
        self.par_map = par_map
        self.samples = samples

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
        par_map = {
            'hello': {
                'paramset': paramset(n_parameters = 2, inits = [0,0], bounds = [[-5,5],[-5,5]]),
                'slice': slice(0,1),
            },
            'world': {
                'paramset': paramset(n_parameters = 2, inits = [0,0], bounds = [[-5,5],[-5,5]]),
                'slice': slice(1,2),
            }
        },
        par_order = ['hello','world'],
        samples = ['signal','background']
    )

    mega_mods = {
        'signal': {
            'histosys/hello': {
                'type': 'histosys',
                'name': 'hello',
                'data': {
                    'hi_data' : [11,12,13],
                    'lo_data' : [9,8,7],
                    'nom_data': [10,10,10],
                    'mask'    : [True,True,True]
                }
            },
            'histosys/world': {
                'type': 'histosys',
                'name': 'world',
                'data': {
                    'hi_data' : [10,10,10],
                    'lo_data' : [5,6,7],
                    'nom_data': [10,10,10],
                    'mask'    : [True,True,True]
                }
            }
        },
        'background': {
            'histosys/hello': {
                'type': 'histosys',
                'name': 'hello',
                'data': {
                    'hi_data' : [11,12,13],
                    'lo_data' : [9,8,7],
                    'nom_data': [10,10,10],
                    'mask'    : [True,True,True]
                }
            },
            'histosys/world': {
                'type': 'histosys',
                'name': 'world',
                'data': {
                    'hi_data' : [10,10,10],
                    'lo_data' : [5,6,7],
                    'nom_data': [10,10,10],
                    'mask'    : [True,True,True]
                }
            }
        }}

    hsc = histosys_combined(
        [('hello','histosys'),('world','histosys')] ,
        mc,
        mega_mods
    )

    mod = hsc.apply(pyhf.tensorlib.astensor([0.5,-1.0]))
    shape = pyhf.tensorlib.shape(mod)
    assert shape == (2,2,1,3)


@pytest.mark.skip_mxnet
def test_normsys(backend):
    mc = MockConfig(
        par_map = {
            'hello': {
                'paramset': paramset(n_parameters = 2, inits = [0,0], bounds = [[-5,5],[-5,5]]),
                'slice': slice(0,1),
            },
            'world': {
                'paramset': paramset(n_parameters = 2, inits = [0,0], bounds = [[-5,5],[-5,5]]),
                'slice': slice(1,2),
            }
        },
        par_order = ['hello','world'],
        samples = ['signal','background']
    )

    mega_mods = {
        'signal': {
            'normsys/hello': {
                'type': 'histosys',
                'name': 'hello',
                'data': {
                    'hi' : [1.1]*3,
                    'lo' : [0.9]*3,
                    'nom_data': [10,10,10],
                    'mask'    : [True,True,True]
                }
            },
            'normsys/world': {
                'type': 'histosys',
                'name': 'world',
                'data': {
                    'hi' : [1.1]*3,
                    'lo' : [0.9]*3,
                    'nom_data': [10,10,10],
                    'mask'    : [True,True,True]
                }
            }
        },
        'background': {
            'normsys/hello': {
                'type': 'histosys',
                'name': 'hello',
                'data': {
                    'hi' : [1.1]*3,
                    'lo' : [0.9]*3,
                    'nom_data': [10,10,10],
                    'mask'    : [True,True,True]
                }
            },
            'normsys/world': {
                'type': 'histosys',
                'name': 'world',
                'data': {
                    'hi' : [1.1]*3,
                    'lo' : [0.9]*3,
                    'nom_data': [10,10,10],
                    'mask'    : [True,True,True]
                }
            }
        }}

    hsc = normsys_combined(
        [('hello','normsys'),('world','normsys')] ,
        mc,
        mega_mods
    )

    mod = hsc.apply(pyhf.tensorlib.astensor([0.5,-1.0]))
    shape = pyhf.tensorlib.shape(mod)
    assert shape == (2,2,1,3)


@pytest.mark.skip_mxnet
def test_lumi(backend):
    mc = MockConfig(
        par_map = {
            'lumi': {
                'paramset': paramset(n_parameters = 1, inits = [0], bounds = [[-5,5]]),
                'slice': slice(0,1),
            },
        },
        par_order = ['lumi'],
        samples = ['signal','background']
    )

    mega_mods = {
        'signal': {
            'lumi/lumi': {
                'type': 'lumi',
                'name': 'lumi',
                'data': {
                    'mask'    : [True,True,True]
                }
            },
        },
        'background': {
            'lumi/lumi': {
                'type': 'lumi',
                'name': 'lumi',
                'data': {
                    'mask'    : [True,True,True]
                }
            },
        }}

    hsc = lumi_combined(
        [('lumi','lumi')] ,
        mc,
        mega_mods
    )

    mod = hsc.apply(pyhf.tensorlib.astensor([0.5]))
    shape = pyhf.tensorlib.shape(mod)
    assert shape == (1,2,1,3)


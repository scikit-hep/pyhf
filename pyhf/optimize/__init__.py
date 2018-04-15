from .opt_scipy import scipy_optimizer
assert scipy_optimizer

try:
    from .opt_pytorch import pytorch_optimizer
    assert pytorch_optimizer
except ImportError:
    pass

try:
    from .opt_tflow import tflow_optimizer
    assert tflow_optimizer
except ImportError:
    pass

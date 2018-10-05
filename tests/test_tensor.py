import pyhf

from pyhf.tensor.numpy_backend import numpy_backend
from pyhf.tensor.tensorflow_backend import tensorflow_backend
from pyhf.tensor.pytorch_backend import pytorch_backend
from pyhf.tensor.mxnet_backend import mxnet_backend
from pyhf.simplemodels import hepdata_like

import numpy as np
import tensorflow as tf

import pytest


def test_tensor_list_conversion(backend):
    tb = pyhf.tensorlib
    assert tb.tolist(tb.astensor([1, 2, 3])) == [1, 2, 3]

def test_simple_tensor_ops(backend):
    tb = pyhf.tensorlib
    assert tb.tolist(tb.sum([[1, 2, 3], [4, 5, 6]], axis=0)) == [5, 7, 9]
    assert tb.tolist(tb.product([[1, 2, 3], [4, 5, 6]], axis=0)) == [4, 10, 18]
    assert tb.tolist(tb.power([1, 2, 3], [1, 2, 3])) == [1, 4, 27]
    assert tb.tolist(tb.divide([4, 9, 16], [2, 3, 4])) == [2, 3, 4]
    assert tb.tolist(tb.sqrt([4, 9, 16])) == [2, 3, 4]
    assert tb.tolist(tb.log(tb.exp([2, 3, 4]))) == [2, 3, 4]
    assert tb.tolist(tb.abs([-1, -2])) == [1, 2]

def test_complex_tensor_ops(backend):
    tb = pyhf.tensorlib
    assert tb.tolist(
        tb.outer([1, 2, 3], [4, 5, 6])) == [[4, 5, 6], [8, 10, 12], [12, 15, 18]]
    assert tb.tolist(tb.stack(
        [tb.astensor([1, 2, 3]), tb.astensor([4, 5, 6])])) == [[1, 2, 3], [4, 5, 6]]
    assert tb.tolist(tb.concatenate(
        [tb.astensor([1, 2, 3]), tb.astensor([4, 5, 6])])) == [1, 2, 3, 4, 5, 6]
    assert tb.tolist(
        tb.clip(tb.astensor([-2, -1, 0, 1, 2]), -1, 1)) == [-1, -1,  0,  1,  1]
    assert tb.tolist(tb.where(
        tb.astensor([1, 0, 1]),
        tb.astensor([1, 1, 1]),
        tb.astensor([2, 2, 2]))) == [1, 2, 1]

def test_zeros(backend):
    tb = pyhf.tensorlib
    assert tb.tolist(tb.ones((2, 3))) == [[1, 1, 1], [1, 1, 1]]
    assert tb.tolist(tb.ones((4, 5))) == [[1.] * 5] * 4

def test_ones(backend):
    tb = pyhf.tensorlib
    assert tb.tolist(tb.zeros((4, 5))) == [[0.] * 5] * 4

def test_broadcasting(backend):
    tb = pyhf.tensorlib
    assert list(map(tb.tolist, tb.simple_broadcast(
        tb.astensor([1, 1, 1]),
        tb.astensor([2]),
        tb.astensor([3, 3, 3])))) == [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
    assert list(map(tb.tolist, tb.simple_broadcast(1, [2, 3, 4], [5, 6, 7]))) \
        == [[1, 1, 1], [2, 3, 4], [5, 6, 7]]
    assert list(map(tb.tolist, tb.simple_broadcast([1], [2, 3, 4], [5, 6, 7]))) \
        == [[1, 1, 1], [2, 3, 4], [5, 6, 7]]
    with pytest.raises(Exception):
        tb.simple_broadcast([1], [2, 3], [5, 6, 7])

def test_reshape(backend):
    tb = pyhf.tensorlib
    assert tb.tolist(tb.reshape(tb.ones((1,2,3)), (-1,))) == [1, 1, 1, 1, 1, 1]

def test_shape(backend):
    tb = pyhf.tensorlib
    assert tb.shape(tb.ones((1,2,3,4,5))) == (1,2,3,4,5)

def test_pdf_calculations(backend):
    tb = pyhf.tensorlib
    assert tb.tolist(
        tb.normal_cdf(tb.astensor([0.8]))) == pytest.approx([0.7881446014166034], 1e-07)
    assert tb.tolist(
        tb.normal_logpdf(tb.astensor([0, 0, 1, 1, 0, 0, 1, 1]), tb.astensor([0, 1, 0, 1, 0, 1, 0, 1]), tb.astensor([0, 0, 0, 0, 1, 1, 1, 1]))) == pytest.approx([np.nan, np.nan, np.nan, np.nan, -0.91893853, -1.41893853, -1.41893853, -0.91893853], nan_ok=True)
    # poisson(lambda=0) is not defined, should return NaN
    assert tb.tolist(
        tb.poisson([0, 0, 1, 1], [0, 1, 0, 1])) == pytest.approx([np.nan, 0.3678794503211975, 0.0, 0.3678794503211975], nan_ok=True)
    assert tb.tolist(
        tb.poisson_logpdf(tb.astensor([0, 0, 1, 1]), tb.astensor([0, 1, 0, 1]))) == pytest.approx(np.log([np.nan, 0.3678794503211975, 0.0, 0.3678794503211975]).tolist(), nan_ok=True)

@pytest.mark.skip_mxnet
def test_boolean_mask(backend):
    tb = pyhf.tensorlib
    assert tb.tolist(tb.boolean_mask(tb.astensor([1,2,3,4,5,6]), tb.astensor([True, True, False, True, False, False], dtype='bool'))) == [1, 2, 4]
    assert tb.tolist(tb.boolean_mask(tb.astensor([[1,2],[3,4],[5,6]]), tb.astensor([[True, True], [False, True], [False, False]], dtype='bool'))) == [1, 2, 4]

def test_1D_gather(backend):
    tb = pyhf.tensorlib
    assert tb.tolist(tb.gather(tb.astensor([1,2,3,4,5,6]), tb.astensor([4,0,3,2], dtype='int'))) == [5, 1, 4, 3]
    assert tb.tolist(tb.gather(tb.astensor([1,2,3,4,5,6]), tb.astensor([[4,0],[3,2]], dtype='int'))) == [[5, 1], [4, 3]]

@pytest.mark.fail_pytorch
def test_ND_gather(backend):
    tb = pyhf.tensorlib
    assert tb.tolist(tb.gather(tb.astensor([[1,2],[3,4],[5,6]]), tb.astensor([1,0], dtype='int'))) == [[3, 4], [1, 2]]

@pytest.mark.fail_mxnet
def test_isfinite(backend):
    tb = pyhf.tensorlib
    assert tb.tolist(tb.isfinite(tb.astensor([1.0, float("nan"), float("inf")]))) == [True, False, False]

def test_einsum(backend):
    tb = pyhf.tensorlib
    x = np.arange(20).reshape(5,4).tolist()

    if isinstance(pyhf.tensorlib, pyhf.tensor.mxnet_backend):
        with pytest.raises(NotImplementedError):
            assert tb.einsum('ij->ji',[1,2,3])
    else:
        assert np.all(tb.tolist(tb.einsum('ij->ji',x)) == np.asarray(x).T.tolist())
        assert tb.tolist(tb.einsum('i,j->ij',tb.astensor([1,1,1]),tb.astensor([1,2,3]))) == [[1,2,3]]*3

@pytest.mark.skip_mxnet
def test_pdf_eval(backend):
    source = {
        "binning": [2, -0.5, 1.5],
        "bindata": {
            "data":    [120.0, 180.0],
            "bkg":     [100.0, 150.0],
            "bkgsys_up":  [102, 190],
            "bkgsys_dn":  [98, 100],
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
                        'modifiers': [{'name': 'mu', 'type': 'normfactor', 'data': None}]
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
    pdf = pyhf.Model(spec)
    data = source['bindata']['data'] + pdf.config.auxdata
    assert pytest.approx([-17.648827643136507], rel=5e-5) == pyhf.tensorlib.tolist(pdf.logpdf(pdf.config.suggested_init(), data))


def test_pdf_eval_2(backend):
    source = {
        "binning": [2, -0.5, 1.5],
        "bindata": {
            "data":    [120.0, 180.0],
            "bkg":     [100.0, 150.0],
            "bkgerr":     [10.0, 10.0],
            "sig":     [30.0, 95.0]
        }
    }

    pdf = hepdata_like(source['bindata']['sig'], source['bindata'][
                       'bkg'], source['bindata']['bkgerr'])
    data = source['bindata']['data'] + pdf.config.auxdata

    assert pytest.approx([-23.579605171119738], rel=5e-5) == pyhf.tensorlib.tolist(pdf.logpdf(pdf.config.suggested_init(), data))

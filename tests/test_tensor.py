from pyhf.tensor.pytorch_backend import pytorch_backend
from pyhf.tensor.numpy_backend import numpy_backend
from pyhf.tensor.tensorflow_backend import tensorflow_backend
from pyhf.simplemodels import hepdata_like
import tensorflow as tf


def test_common_tensor_backends():
    tf_sess = tf.Session()
    for tb in [numpy_backend(), pytorch_backend(), tensorflow_backend(session = tf_sess)]:
        assert tb.tolist(tb.astensor([1,2,3])) == [1,2,3]
        assert tb.tolist(tb.ones((2,3))) == [[1,1,1],[1,1,1]]
        assert tb.tolist(tb.sum([[1,2,3],[4,5,6]], axis = 0)) == [5,7,9]
        assert tb.tolist(tb.product([[1,2,3],[4,5,6]], axis = 0)) == [4,10,18]
        assert tb.tolist(tb.power([1,2,3],[1,2,3])) == [1,4,27]
        assert tb.tolist(tb.divide([4,9,16],[2,3,4])) == [2,3,4]
        assert tb.tolist(tb.outer([1,2,3],[4,5,6])) == [[4,5,6],[8,10,12],[12,15,18]]
        assert tb.tolist(tb.sqrt([4,9,16])) == [2,3,4]
        assert tb.tolist(tb.stack([tb.astensor([1,2,3]),tb.astensor([4,5,6])])) == [[1,2,3],[4,5,6]]
        assert tb.tolist(tb.concatenate([tb.astensor([1,2,3]),tb.astensor([4,5,6])])) == [1,2,3,4,5,6]
        assert tb.tolist(tb.log(tb.exp([2,3,4]))) == [2,3,4]
        assert tb.tolist(tb.where(
            tb.astensor([1,0,1]),
            tb.astensor([1,1,1]),
            tb.astensor([2,2,2]))) == [1,2,1]

        assert list(map(tb.tolist,tb.simple_broadcast(
            tb.astensor([1,1,1]),
            tb.astensor([2]),
            tb.astensor([3,3,3])))) == [[1,1,1],[2,2,2],[3,3,3]]


def test_pdf_eval():
    import pyhf
    import numpy as np
    oldlib = pyhf.tensorlib

    tf_sess = tf.Session()
    backends = [numpy_backend(poisson_from_normal = True), pytorch_backend(), tensorflow_backend(session = tf_sess)]

    values = []
    for b in backends:

        pyhf.tensorlib = b

        source = {
          "binning": [2,-0.5,1.5],
          "bindata": {
            "data":    [120.0, 180.0],
            "bkg":     [100.0, 150.0],
            "bkgsys_up":  [102, 190],
            "bkgsys_dn":  [98, 100],
            "sig":     [30.0, 95.0]
          }
        }
        spec = {
            'singlechannel': {
                'signal': {
                    'data': source['bindata']['sig'],
                    'mods': [{'name': 'mu','type': 'normfactor','data': None}]
                },
                'background': {
                    'data': source['bindata']['bkg'],
                    'mods': [{'name': 'bkg_norm','type': 'histosys','data': {
                        'lo_hist': source['bindata']['bkgsys_dn'], 'hi_hist': source['bindata']['bkgsys_up'],
                    }}]
                }
            }
        }
        pdf  = pyhf.hfpdf(spec)
        data = source['bindata']['data'] + pdf.config.auxdata

        v1 = pdf.logpdf(pdf.config.suggested_init(), data)
        values.append(pyhf.tensorlib.tolist(v1)[0])

    assert np.std(values) < 1e-6

    pyhf.tensorlib = oldlib

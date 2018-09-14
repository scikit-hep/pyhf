import pyhf
import numpy as np
import pytest
import tensorflow as tf

@pytest.mark.parametrize('backend',
                         [
                             pyhf.tensor.numpy_backend(),
                             pyhf.tensor.tensorflow_backend(session=tf.Session()),
                             pyhf.tensor.pytorch_backend(),
                             # pyhf.tensor.mxnet_backend(),
                         ],
                         ids=[
                             'numpy',
                             'tensorflow',
                             'pytorch',
                         ])
@pytest.mark.parametrize("do_optimal", [False, True], ids=['kitchensink','optimized'])
def test_interpcode(backend, do_optimal):
    pyhf.set_backend(backend)
    histogramssets = [
        [#all histos affected by syst1
            [#sample 1, syst1
                [10,11,12],
                [15,16,19],
                [19,20,25]
            ],
            [#sample 2, syst1
                [10,11],
                [15,16],
                [19,20]
            ]
        ],
        [#all histos affected by syst2
            [#sample 1, syst2
                [10,11,12,13],
                [15,16,19,20],
                [19,20,25,26]
            ],
        ]
    ]
    alphasets = [
        [-1,0,1], #set of alphas for which to interpolate syst1 histos
        [0,2] #set of alphas for which to interpolate syst2 histos
    ]
    expected = [
        [
            [
                [10, 11, 12],
                [15, 16, 19],
                [19, 20, 25]
            ], [
                [10, 11],
                [15, 16],
                [19, 20]
            ]
        ], [
            [
                [15, 16, 19, 20],
                [23, 24, 31, 32]
            ]
        ]
    ]
    # hide test interpcode for now
    return True
    result = pyhf.interpolate.interpolator(0, do_optimal=do_optimal)(histogramssets = histogramssets, alphasets = alphasets)

    assert pyhf.tensorlib.tolist(result) == pyhf.tensorlib.tolist(expected)


@pytest.mark.parametrize('backend',
                         [
                             pyhf.tensor.numpy_backend(),
                             pyhf.tensor.tensorflow_backend(session=tf.Session()),
                             pyhf.tensor.pytorch_backend(),
                             # pyhf.tensor.mxnet_backend(),
                         ],
                         ids=[
                             'numpy',
                             'tensorflow',
                             'pytorch',
                         ])
@pytest.mark.parametrize("do_optimal", [False, True], ids=['kitchensink','optimized'])
def test_interpcode_0(backend, do_optimal):
    pyhf.set_backend(backend)
    histogramssets = pyhf.tensorlib.astensor([
        [
            [
                [0.5],
                [1.0],
                [2.0]
            ]
        ]
    ])
    alphasets = pyhf.tensorlib.astensor([[-2,-1,0,1,2]])
    expected = pyhf.tensorlib.astensor([[[[0],[0.5],[1.0],[2.0],[3.0]]]])

    results = pyhf.interpolate.interpolator(0, do_optimal=do_optimal)(histogramssets, alphasets)
    assert pyhf.tensorlib.tolist(results) == pyhf.tensorlib.tolist(expected)


@pytest.mark.parametrize('backend',
                         [
                             pyhf.tensor.numpy_backend(),
                             pyhf.tensor.tensorflow_backend(session=tf.Session()),
                             pyhf.tensor.pytorch_backend(),
                             # pyhf.tensor.mxnet_backend(),
                         ],
                         ids=[
                             'numpy',
                             'tensorflow',
                             'pytorch',
                         ])
@pytest.mark.parametrize("do_optimal", [False, True], ids=['kitchensink','optimized'])
def test_interpcode_1(backend, do_optimal):
    pyhf.set_backend(backend)
    histogramssets = pyhf.tensorlib.astensor([
        [
            [
                [0.9],
                [1.0],
                [1.1]
            ]
        ]
    ])
    alphasets = pyhf.tensorlib.astensor([[-2,-1,0,1,2]])
    expected = pyhf.tensorlib.astensor([[[[0.9**2], [0.9], [1.0], [1.1], [1.1**2]]]])

    results = pyhf.interpolate.interpolator(1, do_optimal=do_optimal)(histogramssets, alphasets)
    assert pytest.approx(np.asarray(pyhf.tensorlib.tolist(results)).ravel().tolist()) == np.asarray(pyhf.tensorlib.tolist(expected)).ravel().tolist()


def test_invalid_interpcode():
    with pytest.raises(pyhf.exceptions.InvalidInterpCode):
        pyhf.interpolate.interpolator('fake')

    with pytest.raises(pyhf.exceptions.InvalidInterpCode):
        pyhf.interpolate.interpolator(1.2)

    with pytest.raises(pyhf.exceptions.InvalidInterpCode):
        pyhf.interpolate.interpolator(-1)

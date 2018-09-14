import pyhf
import numpy as np
import pytest
import tensorflow as tf

@pytest.fixture
def random_histosets_alphasets_pair():
    def generate_shapes(histogramssets,alphasets):
        h_shape = [len(histogramssets),0,0,0]
        a_shape = (len(alphasets),max(map(len,alphasets)))
        for hs in histogramssets:
            h_shape[1] = max(h_shape[1],len(hs))
            for h in hs:
                h_shape[2] = max(h_shape[2],len(h))
                for sh in h:
                    h_shape[3] = max(h_shape[3],len(sh))
        return tuple(h_shape),a_shape

    def filled_shapes(histogramssets,alphasets):
        # pad our shapes with NaNs
        histos, alphas = generate_shapes(histogramssets,alphasets)
        histos, alphas = np.ones(histos) * np.nan, np.ones(alphas) * np.nan
        for i,syst in enumerate(histogramssets):
            for j,sample in enumerate(syst):
                for k,variation in enumerate(sample):
                    histos[i,j,k,:len(variation)] = variation
        for i,alphaset in enumerate(alphasets):
            alphas[i,:len(alphaset)] = alphaset
        return histos,alphas

    nsysts = 150
    nhistos_per_syst_upto = 300
    nalphas = 1
    nbins_upto = 1

    nsyst_histos = np.random.randint(1, 1+nhistos_per_syst_upto, size=nsysts)
    nhistograms = [np.random.randint(1, nbins_upto+1, size=n) for n in nsyst_histos]
    random_alphas = [np.random.uniform(-1, 1,size=nalphas) for n in nsyst_histos]

    random_histogramssets = [
        [# all histos affected by systematic $nh
            [# sample $i, systematic $nh
                np.random.uniform(10*i+j,10*i+j+1, size = nbin).tolist() for j in range(3)
            ] for i,nbin in enumerate(nh)
        ] for nh in nhistograms
    ]
    h,a = filled_shapes(random_histogramssets,random_alphas)
    return h,a

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
@pytest.mark.parametrize("interpcode", [0, 1])
def test_interpcode(backend, interpcode, random_histosets_alphasets_pair):
    pyhf.set_backend(backend)
    histogramssets, alphasets = random_histosets_alphasets_pair

    kitchensink_result = np.asarray(pyhf.tensorlib.tolist(pyhf.interpolate.interpolator(interpcode, do_optimal=False)(histogramssets=histogramssets, alphasets=alphasets)))
    optimized_result = np.asarray(pyhf.tensorlib.tolist(pyhf.interpolate.interpolator(interpcode, do_optimal=True)(histogramssets=pyhf.tensorlib.astensor(histogramssets.tolist()), alphasets=pyhf.tensorlib.astensor(alphasets.tolist()))))

    assert pytest.approx(kitchensink_result[~np.isnan(kitchensink_result)].ravel().tolist()) == optimized_result[~np.isnan(optimized_result)].ravel().tolist()

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

    if do_optimal:
      results = pyhf.interpolate.interpolator(0, do_optimal=do_optimal)(histogramssets, alphasets)
    else:
      results = pyhf.interpolate.interpolator(0, do_optimal=do_optimal)(pyhf.tensorlib.tolist(histogramssets), pyhf.tensorlib.tolist(alphasets))
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

    if do_optimal:
      results = pyhf.interpolate.interpolator(1, do_optimal=do_optimal)(histogramssets, alphasets)
    else:
      results = pyhf.interpolate.interpolator(1, do_optimal=do_optimal)(pyhf.tensorlib.tolist(histogramssets), pyhf.tensorlib.tolist(alphasets))

    assert pytest.approx(np.asarray(pyhf.tensorlib.tolist(results)).ravel().tolist()) == np.asarray(pyhf.tensorlib.tolist(expected)).ravel().tolist()


def test_invalid_interpcode():
    with pytest.raises(pyhf.exceptions.InvalidInterpCode):
        pyhf.interpolate.interpolator('fake')

    with pytest.raises(pyhf.exceptions.InvalidInterpCode):
        pyhf.interpolate.interpolator(1.2)

    with pytest.raises(pyhf.exceptions.InvalidInterpCode):
        pyhf.interpolate.interpolator(-1)

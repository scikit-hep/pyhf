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

def test_badly_defined_interplator(random_histosets_alphasets_pair):
    histogramssets, alphasets = random_histosets_alphasets_pair

    class BadInterpolator(pyhf.interpolate.Interpolator):
        pass

    with pytest.raises(NotImplementedError):
        foo = BadInterpolator(histogramssets)

    class BadInterpolatorToo(pyhf.interpolate.Interpolator):
        def _precompute(self, shape):
            pass

    with pytest.raises(NotImplementedError):
        bar = BadInterpolatorToo(histogramssets)
        bar(alphasets)

@pytest.mark.skip_mxnet
def test_interpolator(backend, random_histosets_alphasets_pair):
    histogramssets, alphasets = random_histosets_alphasets_pair

    interpolator = pyhf.interpolate._hfinterpolator_code0(pyhf.tensorlib.astensor(histogramssets.tolist()))
    assert interpolator.tensorlib == pyhf.tensorlib
    assert interpolator.shape == (histogramssets.shape[0], 1)
    interpolator(pyhf.tensorlib.astensor(alphasets.tolist()))
    assert interpolator.shape == alphasets.shape

@pytest.mark.skip_mxnet
@pytest.mark.parametrize("interpcode", [0, 1])
def test_interpcode(backend, interpcode, random_histosets_alphasets_pair):
    histogramssets, alphasets = random_histosets_alphasets_pair

    # single-float precision backends, calculate using single-floats
    if isinstance(pyhf.tensorlib, pyhf.tensor.tensorflow_backend) or isinstance(pyhf.tensorlib, pyhf.tensor.pytorch_backend):
        histogramssets = np.asarray(histogramssets, dtype=np.float32)
        alphasets = np.asarray(alphasets, dtype=np.float32)

    slow_result = np.asarray(pyhf.interpolate.interpolator(interpcode, do_tensorized_calc=False)(histogramssets=histogramssets, alphasets=alphasets))
    fast_result = np.asarray(pyhf.tensorlib.tolist(pyhf.interpolate.interpolator(interpcode, do_tensorized_calc=True)(histogramssets=pyhf.tensorlib.astensor(histogramssets.tolist()), alphasets=pyhf.tensorlib.astensor(alphasets.tolist()))))

    assert pytest.approx(slow_result[~np.isnan(slow_result)].ravel().tolist()) == fast_result[~np.isnan(fast_result)].ravel().tolist()

@pytest.mark.skip_mxnet
@pytest.mark.parametrize("do_tensorized_calc", [False, True], ids=['slow','fast'])
def test_interpcode_0(backend, do_tensorized_calc):
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

    if do_tensorized_calc:
      result_deltas = pyhf.interpolate.interpolator(0, do_tensorized_calc=do_tensorized_calc)(histogramssets, alphasets)
    else:
      result_deltas = pyhf.tensorlib.astensor(pyhf.interpolate.interpolator(0, do_tensorized_calc=do_tensorized_calc)(pyhf.tensorlib.tolist(histogramssets), pyhf.tensorlib.tolist(alphasets)))


    # calculate the actual change
    allsets_allhistos_noms_repeated = pyhf.tensorlib.einsum('sa,shb->shab', pyhf.tensorlib.ones(alphasets.shape), histogramssets[:,:,1])
    results = allsets_allhistos_noms_repeated + result_deltas

    assert pytest.approx(np.asarray(pyhf.tensorlib.tolist(results)).ravel().tolist()) == np.asarray(pyhf.tensorlib.tolist(expected)).ravel().tolist()

@pytest.mark.skip_mxnet
@pytest.mark.parametrize("do_tensorized_calc", [False, True], ids=['slow','fast'])
def test_interpcode_1(backend, do_tensorized_calc):
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

    if do_tensorized_calc:
      result_deltas = pyhf.interpolate.interpolator(1, do_tensorized_calc=do_tensorized_calc)(histogramssets, alphasets)
    else:
      result_deltas = pyhf.tensorlib.astensor(pyhf.interpolate.interpolator(1, do_tensorized_calc=do_tensorized_calc)(pyhf.tensorlib.tolist(histogramssets), pyhf.tensorlib.tolist(alphasets)))

    # calculate the actual change
    allsets_allhistos_noms_repeated = pyhf.tensorlib.einsum('sa,shb->shab', pyhf.tensorlib.ones(alphasets.shape), histogramssets[:,:,1])
    results = allsets_allhistos_noms_repeated * result_deltas

    assert pytest.approx(np.asarray(pyhf.tensorlib.tolist(results)).ravel().tolist()) == np.asarray(pyhf.tensorlib.tolist(expected)).ravel().tolist()


def test_invalid_interpcode():
    with pytest.raises(pyhf.exceptions.InvalidInterpCode):
        pyhf.interpolate.interpolator('fake')

    with pytest.raises(pyhf.exceptions.InvalidInterpCode):
        pyhf.interpolate.interpolator(1.2)

    with pytest.raises(pyhf.exceptions.InvalidInterpCode):
        pyhf.interpolate.interpolator(-1)

import pyhf
import numpy as np
import pytest
import mock


@pytest.fixture
def random_histosets_alphasets_pair():
    def generate_shapes(histogramssets, alphasets):
        h_shape = [len(histogramssets), 0, 0, 0]
        a_shape = (len(alphasets), max(map(len, alphasets)))
        for hs in histogramssets:
            h_shape[1] = max(h_shape[1], len(hs))
            for h in hs:
                h_shape[2] = max(h_shape[2], len(h))
                for sh in h:
                    h_shape[3] = max(h_shape[3], len(sh))
        return tuple(h_shape), a_shape

    def filled_shapes(histogramssets, alphasets):
        # pad our shapes with NaNs
        histos, alphas = generate_shapes(histogramssets, alphasets)
        histos, alphas = np.ones(histos) * np.nan, np.ones(alphas) * np.nan
        for i, syst in enumerate(histogramssets):
            for j, sample in enumerate(syst):
                for k, variation in enumerate(sample):
                    histos[i, j, k, : len(variation)] = variation
        for i, alphaset in enumerate(alphasets):
            alphas[i, : len(alphaset)] = alphaset
        return histos, alphas

    nsysts = 150
    nhistos_per_syst_upto = 300
    nalphas = 1
    nbins_upto = 1

    nsyst_histos = np.random.randint(1, 1 + nhistos_per_syst_upto, size=nsysts)
    nhistograms = [np.random.randint(1, nbins_upto + 1, size=n) for n in nsyst_histos]
    random_alphas = [np.random.uniform(-1, 1, size=nalphas) for n in nsyst_histos]

    random_histogramssets = [
        [  # all histos affected by systematic $nh
            [  # sample $i, systematic $nh
                np.random.uniform(10 * i + j, 10 * i + j + 1, size=nbin).tolist()
                for j in range(3)
            ]
            for i, nbin in enumerate(nh)
        ]
        for nh in nhistograms
    ]
    h, a = filled_shapes(random_histogramssets, random_alphas)
    return h, a


def test_interpolator_structure(interpcode, random_histosets_alphasets_pair):
    histogramssets, alphasets = random_histosets_alphasets_pair

    interpolator = pyhf.interpolators.get(interpcode)(
        histogramssets.tolist(), subscribe=False
    )
    assert callable(interpolator)
    assert hasattr(interpolator, 'alphasets_shape')
    assert hasattr(interpolator, '_precompute') and callable(interpolator._precompute)
    assert hasattr(interpolator, '_precompute_alphasets') and callable(
        interpolator._precompute_alphasets
    )


def test_interpolator_subscription(interpcode, random_histosets_alphasets_pair):
    histogramssets, alphasets = random_histosets_alphasets_pair
    ename = 'tensorlib_changed'

    # inject into our interpolator class
    interpolator_cls = pyhf.interpolators.get(interpcode)
    with mock.patch('{0:s}._precompute'.format(interpolator_cls.__module__)) as m:
        interpolator_cls(histogramssets.tolist(), subscribe=False)
        assert m.call_count == 1
        assert m not in pyhf.events.__events.get(ename, [])
        pyhf.events.trigger(ename)()
        assert m.call_count == 1

    with mock.patch('{0:s}._precompute'.format(interpolator_cls.__module__)) as m:
        interpolator_cls(histogramssets.tolist(), subscribe=True)
        assert m.call_count == 1
        assert m in pyhf.events.__events.get(ename, [])
        pyhf.events.trigger(ename)()
        assert m.call_count == 2


def test_interpolator_alphaset_change(
    backend, interpcode, random_histosets_alphasets_pair
):
    histogramssets, alphasets = random_histosets_alphasets_pair
    interpolator = pyhf.interpolators.get(interpcode)(
        histogramssets.tolist(), subscribe=False
    )
    # set to None to force recomputation
    interpolator.alphasets_shape = None
    # expect recomputation to not fail
    interpolator._precompute_alphasets(alphasets.shape)
    # make sure it sets the right shape
    assert interpolator.alphasets_shape == alphasets.shape


def test_interpolator(backend, interpcode, random_histosets_alphasets_pair):
    histogramssets, alphasets = random_histosets_alphasets_pair

    interpolator = pyhf.interpolators.get(interpcode)(
        histogramssets.tolist(), subscribe=False
    )
    assert interpolator.alphasets_shape == (histogramssets.shape[0], 1)
    interpolator.alphasets_shape = None
    interpolator(pyhf.tensorlib.astensor(alphasets.tolist()))
    assert interpolator.alphasets_shape == alphasets.shape


def test_validate_implementation(backend, interpcode, random_histosets_alphasets_pair):
    histogramssets, alphasets = random_histosets_alphasets_pair

    # single-float precision backends, calculate using single-floats
    if pyhf.tensorlib.name in ['tensorflow', 'pytorch']:
        abs_tolerance = 1e-6
        histogramssets = np.asarray(histogramssets, dtype=np.float32)
        alphasets = np.asarray(alphasets, dtype=np.float32)
    else:
        abs_tolerance = 1e-12

    histogramssets = histogramssets.tolist()
    alphasets = pyhf.tensorlib.astensor(alphasets.tolist())

    slow_interpolator = pyhf.interpolators.get(interpcode, do_tensorized_calc=False)(
        histogramssets, subscribe=False
    )
    fast_interpolator = pyhf.interpolators.get(interpcode, do_tensorized_calc=True)(
        histogramssets, subscribe=False
    )
    slow_result = np.asarray(pyhf.tensorlib.tolist(slow_interpolator(alphasets)))
    fast_result = np.asarray(pyhf.tensorlib.tolist(fast_interpolator(alphasets)))

    assert slow_result.shape == fast_result.shape

    assert (
        pytest.approx(
            slow_result[~np.isnan(slow_result)].ravel().tolist(), abs=abs_tolerance
        )
        == fast_result[~np.isnan(fast_result)].ravel().tolist()
    )


@pytest.mark.parametrize("do_tensorized_calc", [False, True], ids=['slow', 'fast'])
def test_code0_validation(backend, do_tensorized_calc):
    histogramssets = [[[[0.5], [1.0], [2.0]]]]
    alphasets = pyhf.tensorlib.astensor([[-2, -1, 0, 1, 2]])
    expected = pyhf.tensorlib.astensor([[[[0], [0.5], [1.0], [2.0], [3.0]]]])

    interpolator = pyhf.interpolators.get(0, do_tensorized_calc=do_tensorized_calc)(
        histogramssets, subscribe=False
    )
    result_deltas = pyhf.tensorlib.astensor(interpolator(alphasets))

    # calculate the actual change
    histogramssets = pyhf.tensorlib.astensor(histogramssets)
    allsets_allhistos_noms_repeated = pyhf.tensorlib.einsum(
        'sa,shb->shab',
        pyhf.tensorlib.ones(pyhf.tensorlib.shape(alphasets)),
        histogramssets[:, :, 1],
    )
    results = allsets_allhistos_noms_repeated + result_deltas

    assert (
        pytest.approx(np.asarray(pyhf.tensorlib.tolist(results)).ravel().tolist())
        == np.asarray(pyhf.tensorlib.tolist(expected)).ravel().tolist()
    )


@pytest.mark.parametrize("do_tensorized_calc", [False, True], ids=['slow', 'fast'])
def test_code1_validation(backend, do_tensorized_calc):
    histogramssets = [[[[0.9], [1.0], [1.1]]]]
    alphasets = pyhf.tensorlib.astensor([[-2, -1, 0, 1, 2]])
    expected = pyhf.tensorlib.astensor(
        [[[[0.9 ** 2], [0.9], [1.0], [1.1], [1.1 ** 2]]]]
    )

    interpolator = pyhf.interpolators.get(1, do_tensorized_calc=do_tensorized_calc)(
        histogramssets, subscribe=False
    )
    result_deltas = interpolator(alphasets)

    # calculate the actual change
    histogramssets = pyhf.tensorlib.astensor(histogramssets)
    allsets_allhistos_noms_repeated = pyhf.tensorlib.einsum(
        'sa,shb->shab',
        pyhf.tensorlib.ones(pyhf.tensorlib.shape(alphasets)),
        histogramssets[:, :, 1],
    )
    results = allsets_allhistos_noms_repeated * result_deltas

    assert (
        pytest.approx(np.asarray(pyhf.tensorlib.tolist(results)).ravel().tolist())
        == np.asarray(pyhf.tensorlib.tolist(expected)).ravel().tolist()
    )


def test_invalid_interpcode():
    with pytest.raises(pyhf.exceptions.InvalidInterpCode):
        pyhf.interpolators.get('fake')

    with pytest.raises(pyhf.exceptions.InvalidInterpCode):
        pyhf.interpolators.get(1.2)

    with pytest.raises(pyhf.exceptions.InvalidInterpCode):
        pyhf.interpolators.get(-1)

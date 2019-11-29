from pyhf.parameters import paramsets
import pytest


def test_paramset_unconstrained():
    pset = paramsets.unconstrained(
        n_parameters=5,
        inits=[0, 1, 2, 3, 4],
        bounds=[(-1, 1), (-2, 2), (-3, 3), (-4, 4)],
    )
    assert pset.suggested_init == [0, 1, 2, 3, 4]
    assert pset.suggested_bounds == [(-1, 1), (-2, 2), (-3, 3), (-4, 4)]
    assert not pset.constrained


def test_paramset_constrained_custom_sigmas():
    pset = paramsets.constrained_by_normal(
        n_parameters=5,
        inits=[0, 1, 2, 3, 4],
        bounds=[(-1, 1), (-2, 2), (-3, 3), (-4, 4)],
        auxdata=[0, 0, 0, 0, 0],
        sigmas=[1, 2, 3, 4, 5],
    )
    assert pset.suggested_init == [0, 1, 2, 3, 4]
    assert pset.suggested_bounds == [(-1, 1), (-2, 2), (-3, 3), (-4, 4)]
    assert pset.constrained
    assert pset.width() == [1, 2, 3, 4, 5]


def test_paramset_constrained_default_sigmas():
    pset = paramsets.constrained_by_normal(
        n_parameters=5,
        inits=[0, 1, 2, 3, 4],
        bounds=[(-1, 1), (-2, 2), (-3, 3), (-4, 4)],
        auxdata=[0, 0, 0, 0, 0],
    )
    assert pset.suggested_init == [0, 1, 2, 3, 4]
    assert pset.suggested_bounds == [(-1, 1), (-2, 2), (-3, 3), (-4, 4)]
    assert pset.constrained
    assert pset.width() == [1, 1, 1, 1, 1]


def test_paramset_constrained_custom_factors():
    pset = paramsets.constrained_by_poisson(
        n_parameters=5,
        inits=[0, 1, 2, 3, 4],
        bounds=[(-1, 1), (-2, 2), (-3, 3), (-4, 4)],
        auxdata=[0, 0, 0, 0, 0],
        factors=[100, 400, 900, 1600, 2500],
    )
    assert pset.suggested_init == [0, 1, 2, 3, 4]
    assert pset.suggested_bounds == [(-1, 1), (-2, 2), (-3, 3), (-4, 4)]
    assert pset.constrained
    assert pset.width() == [1 / 10.0, 1 / 20.0, 1 / 30.0, 1 / 40.0, 1 / 50.0]


def test_paramset_constrained_missiing_factors():
    pset = paramsets.constrained_by_poisson(
        n_parameters=5,
        inits=[0, 1, 2, 3, 4],
        bounds=[(-1, 1), (-2, 2), (-3, 3), (-4, 4)],
        auxdata=[0, 0, 0, 0, 0],
        factors=None,
    )
    with pytest.raises(RuntimeError):
        pset.width()

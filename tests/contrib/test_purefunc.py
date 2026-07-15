import json

import numpy as np
import pytest

import pyhf
from pyhf.contrib.extended_modifiers import purefunc


@pytest.fixture
def modifier_set():
    return purefunc.enable()


def test_missing_bindings(datadir, modifier_set):

    with datadir.joinpath("single_func.json").open() as spec_file:
        spec = json.load(spec_file)
    del spec["bindings"]
    pyhf.set_backend("jax")
    with pytest.raises(pyhf.exceptions.InvalidModel):
        pyhf.Model(
            spec,
            modifier_set=modifier_set,
            poi_name="kappa",
            validate=True,
            schema="defs.json",
        )


def test_backend(datadir, modifier_set):
    with datadir.joinpath("single_func.json").open() as spec_file:
        spec = json.load(spec_file)

    with pytest.raises(pyhf.exceptions.InvalidWorkspaceOperation):
        pyhf.Model(
            spec,
            modifier_set=modifier_set,
            poi_name="kappa",
            validate=True,
            schema="defs.json",
        )


def test_single_func(datadir, modifier_set):
    with datadir.joinpath("single_func.json").open() as spec_file:
        spec = json.load(spec_file)
    pyhf.set_backend("jax")

    model = pyhf.Model(
        spec,
        modifier_set=modifier_set,
        poi_name="kappa",
        validate=True,
        schema="defs.json",
    )

    assert model.config.suggested_init() == pytest.approx([1.5])
    assert model.config.suggested_bounds()[0] == pytest.approx([0.0, 12.0])
    observation = [10, 2]
    inferred = pyhf.infer.mle.fit(data=observation, pdf=model)

    assert pytest.approx(np.sqrt(2), rel=1e-3) == inferred[0]


def test_multi_channel(datadir, modifier_set):
    with datadir.joinpath("two_channels.json").open() as spec_file:
        spec = json.load(spec_file)
    pyhf.set_backend("jax")

    model = pyhf.Model(
        spec,
        modifier_set=modifier_set,
        poi_name="kappa",
        validate=True,
        schema="defs.json",
    )

    assert len(model.config.parameters) == 3
    bounds = np.array(model.config.suggested_bounds())
    alpha_idx = model.config.par_slice("alpha")
    theta_idx = model.config.par_slice("theta")
    kappa_idx = model.config.par_slice("kappa")

    assert np.all(np.isclose(bounds[alpha_idx], [[2.0, 10.0]]))
    assert np.all(np.isclose(bounds[theta_idx], [[0.0, 10.0]]))
    assert np.all(np.isclose(bounds[kappa_idx], [[0.0, 10.0]]))

    observation = [28, 92, 20, 92, 2]
    inferred = pyhf.infer.mle.fit(data=observation, pdf=model)
    assert inferred[alpha_idx] == pytest.approx(4.0, rel=1e-3)
    assert inferred[theta_idx] == pytest.approx(5.0, rel=1e-3)
    assert inferred[kappa_idx] == pytest.approx(2.0, rel=1e-3)


def test_language(datadir, modifier_set):
    with datadir.joinpath("single_func.json").open() as spec_file:
        spec = json.load(spec_file)
    spec["bindings"][0]["language"] = "not_sympy"
    pyhf.set_backend("jax")

    with pytest.raises(purefunc.InvalidLanguage):
        pyhf.Model(
            spec,
            modifier_set=modifier_set,
            poi_name="kappa",
            validate=True,
            schema="defs.json",
        )


def test_backward_bindings(datadir, modifier_set):
    with datadir.joinpath("backward_binding.json").open() as spec_file:
        spec = json.load(spec_file)
    pyhf.set_backend("jax")
    model = pyhf.Model(
        spec,
        modifier_set=modifier_set,
        poi_name="kappa",
        validate=True,
        schema="defs.json",
    )

    assert set(model.config.parameters) == {"kappa", "theta"}


def test_forward_bindings(datadir, modifier_set):
    with datadir.joinpath("backward_binding.json").open() as spec_file:
        spec = json.load(spec_file)
    pyhf.set_backend("jax")
    model = pyhf.Model(
        spec,
        modifier_set=modifier_set,
        poi_name="kappa",
        validate=True,
        schema="defs.json",
    )

    assert set(model.config.parameters) == {"kappa", "theta"}


def test_circular_bindings(datadir, modifier_set):
    with datadir.joinpath("circular_binding.json").open() as spec_file:
        spec = json.load(spec_file)
    pyhf.set_backend("jax")
    with pytest.raises(purefunc.InvalidExpression):
        pyhf.Model(
            spec,
            modifier_set=modifier_set,
            poi_name="kappa",
            validate=True,
            schema="defs.json",
        )

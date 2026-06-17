import pytest
import numpy as np
import pyhf
from pyhf.contrib.extended_modifiers import purefunc
import json

def test_missing_func(datadir, reset_backend):
    reset_backend

    modifier_set = purefunc.enable()
    with datadir.joinpath("single_func.json").open() as spec_file:
        spec = json.load(spec_file)
    del spec['functions']
    with pytest.raises(pyhf.exceptions.InvalidModel):
        model = pyhf.Model(
            spec,
            modifier_set=modifier_set, 
            poi_name="kappa", validate=True, schema="defs.json")

def test_backend(datadir, reset_backend):

    # what to test:
    # exception if backend isnt jax
    # exception if function isnt declared
    # func shared across samples
    # func shared across channels
    # func sharing parameters
    reset_backend
    with datadir.joinpath("single_func.json").open() as spec_file:
        spec = json.load(spec_file)

    modifier_set = purefunc.enable()
    with pytest.raises(pyhf.exceptions.InvalidWorkspaceOperation):
        model = pyhf.Model(spec, modifier_set=modifier_set, poi_name="kappa", validate=True, schema="defs.json")
    pyhf.set_backend("jax")

    model = pyhf.Model(spec, poi_name="kappa", validate=True, schema="defs.json")
    assert model

def test_single_func(datadir, reset_backend):
    reset_backend
    with datadir.joinpath("single_func.json").open() as spec_file:
        spec = json.load(spec_file)

    modifier_set = purefunc.enable()
    with pytest.raises(pyhf.exceptions.InvalidWorkspaceOperation):
        model = pyhf.Model(spec, modifier_set=modifier_set,poi_name="kappa", validate=True, schema="defs.json")
    pyhf.set_backend("jax")

    model = pyhf.Model(spec, modifier_set=modifier_set,poi_name="kappa", validate=True, schema="defs.json")

    assert model.config.suggested_init() == [1.5]
    assert model.config.suggested_bounds() == [[0., 12.]]
    pars = np.array(model.config.suggested_init())

    observation = [24, 24]

    inferred = pyhf.infer.mle.fit(data=observation, pdf=model)
    print(inferred)
import json

import numpy
import pytest
from jsonpatch import JsonPatch

import pyhf


def test_shapefactor_build():
    spec = {
        'channels': [
            {
                'name': 'channel',
                'samples': [
                    {
                        'name': 'sample',
                        'data': [10.0] * 3,
                        'modifiers': [
                            {'name': 'mu', 'type': 'normfactor', 'data': None},
                        ],
                    },
                    {
                        'name': 'another_sample',
                        'data': [5.0] * 3,
                        'modifiers': [
                            {'name': 'freeshape', 'type': 'shapefactor', 'data': None}
                        ],
                    },
                ],
            }
        ],
    }

    model = pyhf.Model(spec)
    assert model


def test_staterror_holes():
    spec = {
        'channels': [
            {
                'name': 'channel1',
                'samples': [
                    {
                        'name': 'another_sample',
                        'data': [50, 0, 0, 70],
                        'modifiers': [
                            {'name': 'mu', 'type': 'normfactor', 'data': None},
                            {
                                'name': 'staterror_1',
                                'type': 'staterror',
                                'data': [5, 0, 5, 5],
                            },
                        ],
                    },
                ],
            },
            {
                'name': 'channel2',
                'samples': [
                    {
                        'name': 'another_sample',
                        'data': [50, 0, 10, 70],
                        'modifiers': [
                            {
                                'name': 'staterror_2',
                                'type': 'staterror',
                                'data': [5, 0, 5, 5],
                            }
                        ],
                    },
                ],
            },
        ],
    }

    model = pyhf.Model(spec, poi_name="")
    assert model.config.npars == 9
    _, factors = model._modifications(
        pyhf.tensorlib.astensor([2, 2.0, 1.0, 1.0, 3.0, 4.0, 1.0, 5.0, 6.0])
    )
    assert model.config.param_set("staterror_1").suggested_fixed == [
        False,
        True,
        True,
        False,
    ]
    assert all(
        [
            isinstance(fixed, bool)
            for fixed in model.config.param_set("staterror_1").suggested_fixed
        ]
    )
    assert model.config.param_set("staterror_2").suggested_fixed == [
        False,
        True,
        False,
        False,
    ]
    assert all(
        [
            isinstance(fixed, bool)
            for fixed in model.config.param_set("staterror_2").suggested_fixed
        ]
    )
    assert (factors[1][0, 0, 0, :] == [2.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0]).all()
    assert (factors[1][1, 0, 0, :] == [1.0, 1.0, 1.0, 1.0, 4.0, 1.0, 5.0, 6.0]).all()

    data = model.expected_data(model.config.suggested_init())
    assert numpy.isfinite(model.logpdf(model.config.suggested_init(), data)).all()


def test_shapesys_holes():
    spec = {
        'channels': [
            {
                'name': 'channel1',
                'samples': [
                    {
                        'name': 'another_sample',
                        'data': [50, 60, 0, 70],
                        'modifiers': [
                            {'name': 'mu', 'type': 'normfactor', 'data': None},
                            {
                                'name': 'freeshape1',
                                'type': 'shapesys',
                                'data': [5, 0, 5, 5],
                            },
                        ],
                    },
                ],
            },
            {
                'name': 'channel2',
                'samples': [
                    {
                        'name': 'another_sample',
                        'data': [50, 60, 0, 70],
                        'modifiers': [
                            {
                                'name': 'freeshape2',
                                'type': 'shapesys',
                                'data': [5, 0, 5, 5],
                            }
                        ],
                    },
                ],
            },
        ],
    }

    model = pyhf.Model(spec, poi_name="mu")
    _, factors = model._modifications(
        pyhf.tensorlib.astensor([1.0, 2.0, 1.0, 1.0, 3.0, 4.0, 1.0, 1.0, 5.0])
    )
    assert (factors[1][0, 0, 0, :] == [2.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0]).all()
    assert (factors[1][1, 0, 0, :] == [1.0, 1.0, 1.0, 1.0, 4.0, 1.0, 1.0, 5.0]).all()

    assert model.config.param_set("freeshape1").suggested_fixed == [
        False,
        True,
        True,
        False,
    ]
    assert model.config.param_set("freeshape2").suggested_fixed == [
        False,
        True,
        True,
        False,
    ]


@pytest.mark.parametrize(
    "patch_file",
    [
        "bad_histosys_modifier_patch.json",
        "bad_shapesys_modifier_patch.json",
        "bad_staterror_modifier_patch.json",
    ],
)
def test_invalid_bin_wise_modifier(datadir, patch_file):
    """
    Test that bin-wise modifiers will raise an exception if their data shape
    differs from their sample's.
    """
    with open(datadir.joinpath("spec.json"), encoding="utf-8") as spec_file:
        spec = json.load(spec_file)

    assert pyhf.Model(spec)

    with open(datadir.joinpath(patch_file), encoding="utf-8") as read_file:
        patch = JsonPatch.from_string(read_file.read())
    bad_spec = patch.apply(spec)

    with pytest.raises(pyhf.exceptions.InvalidModifier):
        pyhf.Model(bad_spec)


def test_issue1720_staterror_builder_mask(datadir):
    with open(
        datadir.joinpath("issue1720_greedy_staterror.json"), encoding="utf-8"
    ) as spec_file:
        spec = json.load(spec_file)

    spec["channels"][0]["samples"][1]["modifiers"][0]["type"] = "staterror"
    config = pyhf.pdf._ModelConfig(spec)
    builder = pyhf.modifiers.staterror.staterror_builder(config)

    channel = spec["channels"][0]
    sig_sample = channel["samples"][0]
    bkg_sample = channel["samples"][1]
    modifier = bkg_sample["modifiers"][0]

    assert channel["name"] == "channel"
    assert sig_sample["name"] == "signal"
    assert bkg_sample["name"] == "bkg"
    assert modifier["type"] == "staterror"

    builder.append("staterror/NP", "channel", "bkg", modifier, bkg_sample)
    collected_bkg = builder.collect(modifier, bkg_sample["data"])
    assert collected_bkg == {"mask": [True], "nom_data": [1], "uncrt": [1.5]}

    builder.append("staterror/NP", "channel", "signal", None, sig_sample)
    collected_sig = builder.collect(None, sig_sample["data"])
    assert collected_sig == {"mask": [False], "nom_data": [5], "uncrt": [0.0]}

    finalized = builder.finalize()
    assert finalized["staterror/NP"]["bkg"]["data"]["mask"].tolist() == [True]
    assert finalized["staterror/NP"]["signal"]["data"]["mask"].tolist() == [False]


@pytest.mark.parametrize(
    "inits",
    [[-2.0], [-1.0], [0.0], [1.0], [2.0]],
)
def test_issue1720_greedy_staterror(datadir, inits):
    """
    Test that the staterror does not affect more samples than shapesys equivalently.
    """
    with open(
        datadir.joinpath("issue1720_greedy_staterror.json"), encoding="utf-8"
    ) as spec_file:
        spec = json.load(spec_file)

    model_shapesys = pyhf.Workspace(spec).model()
    expected_shapesys = model_shapesys.expected_actualdata(inits)

    spec["channels"][0]["samples"][1]["modifiers"][0]["type"] = "staterror"
    model_staterror = pyhf.Workspace(spec).model()
    expected_staterror = model_staterror.expected_actualdata(inits)

    assert expected_staterror == expected_shapesys

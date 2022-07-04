import pyhf
from jax import numpy as jnp
import jax


def test_model_building_grad():
    pyhf.set_backend('jax', default=True)

    def make_model(
        nominal,
        corrup_data,
        corrdn_data,
        stater_data,
        normsys_up,
        normsys_dn,
        uncorr_data,
    ):
        return {
            "channels": [
                {
                    "name": "achannel",
                    "samples": [
                        {
                            "name": "background",
                            "data": nominal,
                            "modifiers": [
                                {"name": "mu", "type": "normfactor", "data": None},
                                {"name": "lumi", "type": "lumi", "data": None},
                                {
                                    "name": "mod_name",
                                    "type": "shapefactor",
                                    "data": None,
                                },
                                {
                                    "name": "corr_bkguncrt2",
                                    "type": "histosys",
                                    "data": {
                                        'hi_data': corrup_data,
                                        'lo_data': corrdn_data,
                                    },
                                },
                                {
                                    "name": "staterror2",
                                    "type": "staterror",
                                    "data": stater_data,
                                },
                                {
                                    "name": "norm",
                                    "type": "normsys",
                                    "data": {'hi': normsys_up, 'lo': normsys_dn},
                                },
                            ],
                        }
                    ],
                },
                {
                    "name": "secondchannel",
                    "samples": [
                        {
                            "name": "background",
                            "data": nominal,
                            "modifiers": [
                                {"name": "mu", "type": "normfactor", "data": None},
                                {"name": "lumi", "type": "lumi", "data": None},
                                {
                                    "name": "mod_name",
                                    "type": "shapefactor",
                                    "data": None,
                                },
                                {
                                    "name": "uncorr_bkguncrt2",
                                    "type": "shapesys",
                                    "data": uncorr_data,
                                },
                                {
                                    "name": "corr_bkguncrt2",
                                    "type": "histosys",
                                    "data": {
                                        'hi_data': corrup_data,
                                        'lo_data': corrdn_data,
                                    },
                                },
                                {
                                    "name": "staterror",
                                    "type": "staterror",
                                    "data": stater_data,
                                },
                                {
                                    "name": "norm",
                                    "type": "normsys",
                                    "data": {'hi': normsys_up, 'lo': normsys_dn},
                                },
                            ],
                        }
                    ],
                },
            ],
        }

    def pipe(x):
        spec = make_model(
            x * jnp.asarray([60.0, 62.0]),
            x * jnp.asarray([60.0, 62.0]),
            x * jnp.asarray([60.0, 62.0]),
            x * jnp.asarray([5.0, 5.0]),
            x * jnp.asarray(0.95),
            x * jnp.asarray(1.05),
            x * jnp.asarray([5.0, 5.0]),
        )
        model = pyhf.Model(spec, validate=False)
        nominal = jnp.array(model.config.suggested_init())
        data = model.expected_data(nominal)
        return model.logpdf(nominal, data)[0]

    jax.grad(pipe)(3.4)  # should work without error

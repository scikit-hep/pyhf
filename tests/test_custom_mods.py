import pyhf
import pyhf.exceptions as exceptions
import pytest


class custom_builder:
    def __init__(self, pdfconfig):
        self.config = pdfconfig
        self.required_parsets = {
            'k1': [
                {
                    'paramset_type': 'unconstrained',
                    'n_parameters': 1,
                    'is_constrained': False,
                    'is_shared': True,
                    'inits': (1.0,),
                    'bounds': ((-5, 5),),
                    'fixed': False,
                }
            ]
        }
        self.builder_data = {}

    def append(self, key, channel, sample, thismod, defined_samp):
        pass

    def finalize(self):
        return self.builder_data


class custom_applicator:
    op_code = 'multiplication'
    name = 'customfunc'

    def __init__(
        self, modifiers=None, pdfconfig=None, builder_data=None, batch_size=None
    ):
        pass

    def apply(self, pars):
        raise NotImplementedError


def test_custom_mods():
    modifier_set = {custom_applicator.name: (custom_builder, custom_applicator)}
    modifier_set.update(**pyhf.modifiers.histfactory_set)

    model = pyhf.Model(
        {
            'channels': [
                {
                    'name': 'singlechannel',
                    'samples': [
                        {
                            'name': 'signal',
                            'data': [10] * 20,
                            'modifiers': [
                                {
                                    'name': 'singlemod',
                                    'type': 'customfunc',
                                    'data': None,
                                },
                            ],
                        },
                        {'name': 'background', 'data': [300] * 20, 'modifiers': []},
                    ],
                }
            ]
        },
        modifier_set=modifier_set,
        poi_name='k1',
        validate=False,
    )
    assert model
    assert 'k1' in model.config.parameters


def test_missing_poi():
    modifier_set = {custom_applicator.name: (custom_builder, custom_applicator)}
    modifier_set.update(**pyhf.modifiers.histfactory_set)

    with pytest.raises(exceptions.InvalidModel):
        model = pyhf.Model(
            {
                'channels': [
                    {
                        'name': 'singlechannel',
                        'samples': [
                            {
                                'name': 'signal',
                                'data': [10] * 20,
                                'modifiers': [
                                    {
                                        'name': 'singlemod',
                                        'type': 'customfunc',
                                        'data': None,
                                    },
                                ],
                            },
                            {'name': 'background', 'data': [300] * 20, 'modifiers': []},
                        ],
                    }
                ]
            },
            modifier_set=modifier_set,
            poi_name='non_existent_poi',
            validate=False,
        )
        assert model

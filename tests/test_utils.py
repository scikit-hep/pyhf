import pytest
import pyhf


@pytest.mark.parametrize(
    'schema', ['defs.json', 'measurement.json', 'model.json', 'workspace.json']
)
def test_get_schema(schema):
    assert pyhf.utils.load_schema(schema)


def test_load_missing_schema():
    with pytest.raises(IOError):
        pyhf.utils.load_schema('fake_schema.json')


@pytest.mark.parametrize(
    'opts,obj',
    [
        (['a=10'], {'a': 10}),
        (['b=test'], {'b': 'test'}),
        (['c=1.0e-8'], {'c': 1.0e-8}),
        (['d=3.14'], {'d': 3.14}),
        (['e=True'], {'e': True}),
        (['f=false'], {'f': False}),
        (['a=b', 'c=d'], {'a': 'b', 'c': 'd'}),
        (['g=h=i'], {'g': 'h=i'}),
    ],
)
def test_options_from_eqdelimstring(opts, obj):
    assert pyhf.utils.options_from_eqdelimstring(opts) == obj

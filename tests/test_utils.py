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


@pytest.mark.parametrize(
    'obj',
    [
        {'a': 2.0, 'b': 1.0, 'c': 'a'},
        {'b': 1.0, 'c': 'a', 'a': 2.0},
        {'c': 'a', 'a': 2.0, 'b': 1.0},
    ],
)
@pytest.mark.parametrize('algorithm', ['md5', 'sha256'])
def test_digest(obj, algorithm):
    results = {
        'md5': '155e52b05179a1106d71e5e053452517',
        'sha256': '03dfbceade79855fc9b4e4d6fbd4f437109de68330dab37c3091a15f4bffe593',
    }
    assert pyhf.utils.digest(obj, algorithm=algorithm) == results[algorithm]


def test_digest_bad_obj():
    with pytest.raises(ValueError) as excinfo:
        pyhf.utils.digest(object())
    assert 'not JSON-serializable' in str(excinfo.value)


def test_digest_bad_alg():
    with pytest.raises(ValueError) as excinfo:
        pyhf.utils.digest({}, algorithm='nonexistent_algorithm')
    assert 'nonexistent_algorithm' in str(excinfo.value)


@pytest.mark.parametrize(
    'expression',
    [
        '1+2',
        'parameter-name',
        'parameter.name',
        'parameter[name]',
        'parameter["name"]',
        'parameter[:"name"]',
        'parameter[1].a',
        'parameter[1][2]',
        'parameter[1,2]',
        'parameter[1]*2',
        'parameter[-1:5]',
    ],
)
def test_parse_parameter_name_fail(expression):
    with pytest.raises(ValueError):
        pyhf.utils.parse_parameter_name(expression)


@pytest.mark.parametrize(
    'expression,expected',
    [
        ('parameter', slice(None)),
        ('parameter[:]', slice(None)),
        ('parameter[1]', 1),
        ('parameter[1:2]', slice(1, 2)),
        ('parameter[:2]', slice(None, 2)),
        ('parameter[3:]', slice(3, None)),
        ('parameter[4::2]', slice(4, None, 2)),
        ('parameter[::3]', slice(None, None, 3)),
    ],
)
def test_parse_parameter_name(expression, expected):
    pyhf.utils.parse_parameter_name(expression) == expected

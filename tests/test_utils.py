import pyhf
import pytest
import os

def test_get_default_schema():
    assert os.path.isfile(pyhf.utils.get_default_schema())

def test_load_default_schema():
    assert pyhf.utils.load_schema(pyhf.utils.get_default_schema())

def test_load_missing_schema():
    with pytest.raises(IOError):
        pyhf.utils.load_schema('a/fake/path/that/should/not/work.json')

def test_load_custom_schema(tmpdir):
    temp = tmpdir.join("custom_schema.json")
    temp.write('{"foo": "bar"}')
    assert pyhf.utils.load_schema(temp.strpath)

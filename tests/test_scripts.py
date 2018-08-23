import pytest
import json
import shlex

import pyhf

# see test_import.py for the same (detailed) test
def test_import_prepHistFactory(tmpdir, script_runner):
    temp = tmpdir.join("parsed_output.json")
    command = 'pyhf_xml2json --entrypoint-xml validation/xmlimport_input/config/example.xml --basedir validation/xmlimport_input/ --output-file {0:s} --no-tqdm'.format(temp.strpath)
    ret = script_runner.run(*shlex.split(command))
    assert ret.success
    assert ret.stdout == ''
    assert ret.stderr == ''

    parsed_xml = json.loads(temp.read())
    spec = {'channels': parsed_xml['channels']}
    pyhf.utils.validate(spec, pyhf.utils.get_default_schema())

def test_import_prepHistFactory_TQDM(tmpdir, script_runner):
    temp = tmpdir.join("parsed_output.json")
    command = 'pyhf_xml2json --entrypoint-xml validation/xmlimport_input/config/example.xml --basedir validation/xmlimport_input/ --output-file {0:s}'.format(temp.strpath)
    ret = script_runner.run(*shlex.split(command))
    assert ret.success
    assert ret.stdout == ''
    assert ret.stderr != ''


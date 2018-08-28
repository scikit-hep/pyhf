import pytest
import json
import shlex

import pyhf

# see test_import.py for the same (detailed) test
def test_import_prepHistFactory(tmpdir, script_runner):
    temp = tmpdir.join("parsed_output.json")
    command = 'pyhf xml2json validation/xmlimport_input/config/example.xml --basedir validation/xmlimport_input/ --output-file {0:s} --hide-progress'.format(temp.strpath)
    ret = script_runner.run(*shlex.split(command))
    assert ret.success
    assert ret.stdout == ''
    assert ret.stderr == ''

    parsed_xml = json.loads(temp.read())
    spec = {'channels': parsed_xml['channels']}
    pyhf.utils.validate(spec, pyhf.utils.get_default_schema())

def test_import_prepHistFactory_withProgress(tmpdir, script_runner):
    temp = tmpdir.join("parsed_output.json")
    command = 'pyhf xml2json validation/xmlimport_input/config/example.xml --basedir validation/xmlimport_input/ --output-file {0:s}'.format(temp.strpath)
    ret = script_runner.run(*shlex.split(command))
    assert ret.success
    assert ret.stdout == ''
    assert ret.stderr != ''

def test_import_prepHistFactory_stdout(tmpdir, script_runner):
    temp = tmpdir.join("parsed_output.json")
    command = 'pyhf xml2json validation/xmlimport_input/config/example.xml --basedir validation/xmlimport_input/'.format(temp.strpath)
    ret = script_runner.run(*shlex.split(command))
    assert ret.success
    assert ret.stdout != ''
    assert ret.stderr != ''
    d = json.loads(ret.stdout)
    assert d
    assert 'channels' in d

def test_import_prepHistFactory_and_cls(tmpdir, script_runner):
    temp = tmpdir.join("parsed_output.json")
    command = 'pyhf xml2json validation/xmlimport_input/config/example.xml --basedir validation/xmlimport_input/ --output-file {0:s}'.format(temp.strpath)
    ret = script_runner.run(*shlex.split(command))

    command = 'pyhf cls {0:s}'.format(temp.strpath)
    ret = script_runner.run(*shlex.split(command))

    assert ret.success
    d = json.loads(ret.stdout)
    assert d
    assert 'CLs_obs' in d
    assert 'CLs_exp' in d


def test_import_and_export(tmpdir, script_runner):
    temp = tmpdir.join("parsed_output.json")
    command = 'pyhf xml2json validation/xmlimport_input/config/example.xml --basedir validation/xmlimport_input/ --output-file {0:s}'.format(temp.strpath)
    ret = script_runner.run(*shlex.split(command))

    command = 'pyhf json2xml {0:s} --specroot {1:s} --dataroot {1:s}'.format(temp.strpath,str(tmpdir))
    ret = script_runner.run(*shlex.split(command))

import json
import shlex
import pyhf

# see test_import.py for the same (detailed) test
def test_import_prepHistFactory(tmpdir, script_runner):
    temp = tmpdir.join("parsed_output.json")
    command = 'pyhf xml2json validation/xmlimport_input/config/example.xml --basedir validation/xmlimport_input/ --output-file {0:s} --hide-progress'.format(
        temp.strpath
    )
    ret = script_runner.run(*shlex.split(command))
    assert ret.success
    assert ret.stdout == ''
    assert ret.stderr == ''

    parsed_xml = json.loads(temp.read())
    spec = {'channels': parsed_xml['channels']}
    pyhf.utils.validate(spec, pyhf.utils.get_default_schema())


def test_import_prepHistFactory_withProgress(tmpdir, script_runner):
    temp = tmpdir.join("parsed_output.json")
    command = 'pyhf xml2json validation/xmlimport_input/config/example.xml --basedir validation/xmlimport_input/ --output-file {0:s}'.format(
        temp.strpath
    )
    ret = script_runner.run(*shlex.split(command))
    assert ret.success
    assert ret.stdout == ''
    assert ret.stderr != ''


def test_import_prepHistFactory_stdout(tmpdir, script_runner):
    temp = tmpdir.join("parsed_output.json")
    command = 'pyhf xml2json validation/xmlimport_input/config/example.xml --basedir validation/xmlimport_input/'.format(
        temp.strpath
    )
    ret = script_runner.run(*shlex.split(command))
    assert ret.success
    assert ret.stdout != ''
    assert ret.stderr != ''
    d = json.loads(ret.stdout)
    assert d
    assert 'channels' in d


def test_import_prepHistFactory_and_cls(tmpdir, script_runner):
    temp = tmpdir.join("parsed_output.json")
    command = 'pyhf xml2json validation/xmlimport_input/config/example.xml --basedir validation/xmlimport_input/ --output-file {0:s}'.format(
        temp.strpath
    )
    ret = script_runner.run(*shlex.split(command))

    command = 'pyhf cls {0:s}'.format(temp.strpath)
    ret = script_runner.run(*shlex.split(command))

    assert ret.success
    d = json.loads(ret.stdout)
    assert d
    assert 'CLs_obs' in d
    assert 'CLs_exp' in d

    for measurement in [
        'GaussExample',
        'GammaExample',
        'LogNormExample',
        'ConstExample',
    ]:
        command = 'pyhf cls {0:s} --measurement {1:s}'.format(temp.strpath, measurement)
        ret = script_runner.run(*shlex.split(command))

        assert ret.success
        d = json.loads(ret.stdout)
        assert d
        assert 'CLs_obs' in d
        assert 'CLs_exp' in d

        tmp_out = tmpdir.join('{0:s}_output.json'.format(measurement))
        # make sure output file works too
        command += ' --output-file {0:s}'.format(tmp_out.strpath)
        ret = script_runner.run(*shlex.split(command))
        assert ret.success
        d = json.load(tmp_out)
        assert 'CLs_obs' in d
        assert 'CLs_exp' in d


def test_import_and_export(tmpdir, script_runner):
    temp = tmpdir.join("parsed_output.json")
    command = 'pyhf xml2json validation/xmlimport_input/config/example.xml --basedir validation/xmlimport_input/ --output-file {0:s}'.format(
        temp.strpath
    )
    ret = script_runner.run(*shlex.split(command))

    command = 'pyhf json2xml {0:s} --specroot {1:s} --dataroot {1:s}'.format(
        temp.strpath, str(tmpdir)
    )
    ret = script_runner.run(*shlex.split(command))
    assert ret.success


def test_patch(tmpdir, script_runner):
    patch = tmpdir.join('patch.json')

    patchcontent = u'''
[{"op": "replace", "path": "/channels/0/samples/0/data", "value": [5,6]}]
    '''
    patch.write(patchcontent)

    temp = tmpdir.join("parsed_output.json")
    command = 'pyhf xml2json validation/xmlimport_input/config/example.xml --basedir validation/xmlimport_input/ --output-file {0:s}'.format(
        temp.strpath
    )
    ret = script_runner.run(*shlex.split(command))

    command = 'pyhf cls {0:s} --patch {1:s}'.format(temp.strpath, patch.strpath)
    ret = script_runner.run(*shlex.split(command))
    assert ret.success
    import io

    command = 'pyhf cls {0:s} --patch -'.format(temp.strpath, patch.strpath)

    pipefile = io.StringIO(
        patchcontent
    )  # python 2.7 pytest-files are not file-like enough
    ret = script_runner.run(*shlex.split(command), stdin=pipefile)
    print(ret.stderr)
    assert ret.success


def test_patch_fail(tmpdir, script_runner):
    patch = tmpdir.join('patch.json')

    patch.write('''not,json''')

    temp = tmpdir.join("parsed_output.json")
    command = 'pyhf xml2json validation/xmlimport_input/config/example.xml --basedir validation/xmlimport_input/ --output-file {0:s}'.format(
        temp.strpath
    )
    ret = script_runner.run(*shlex.split(command))

    command = 'pyhf cls {0:s} --patch {1:s}'.format(temp.strpath, patch.strpath)
    ret = script_runner.run(*shlex.split(command))
    assert not ret.success


def test_bad_measurement_name(tmpdir, script_runner):
    temp = tmpdir.join("parsed_output.json")
    command = 'pyhf xml2json validation/xmlimport_input/config/example.xml --basedir validation/xmlimport_input/ --output-file {0:s}'.format(
        temp.strpath
    )
    ret = script_runner.run(*shlex.split(command))

    command = 'pyhf cls {0:s} --measurement "a-fake-measurement-name"'.format(
        temp.strpath
    )
    ret = script_runner.run(*shlex.split(command))
    assert not ret.success
    # assert 'no measurement by name' in ret.stderr  # numpy swallows the log.error() here, dunno why

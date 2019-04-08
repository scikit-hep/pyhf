import json
import shlex
import pyhf
import time


def test_version(script_runner):
    command = 'pyhf --version'
    start = time.time()
    ret = script_runner.run(*shlex.split(command))
    end = time.time()
    elapsed = end - start
    assert ret.success
    assert pyhf.__version__ in ret.stdout
    assert ret.stderr == ''
    # make sure it took less than a second
    assert elapsed < 1.0


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

    command = 'pyhf json2xml {0:s} --output-dir {1:s}'.format(
        temp.strpath, tmpdir.mkdir('output').strpath
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


def test_testpoi(tmpdir, script_runner):
    temp = tmpdir.join("parsed_output.json")
    command = 'pyhf xml2json validation/xmlimport_input/config/example.xml --basedir validation/xmlimport_input/ --output-file {0:s}'.format(
        temp.strpath
    )
    ret = script_runner.run(*shlex.split(command))

    pois = [1.0, 0.5, 0.0]
    results_exp = []
    results_obs = []
    for testpoi in pois:
        command = 'pyhf cls {0:s} --testpoi {testpoi:f}'.format(
            temp.strpath, testpoi=testpoi
        )
        ret = script_runner.run(*shlex.split(command))

        assert ret.success
        d = json.loads(ret.stdout)
        assert d
        assert 'CLs_obs' in d
        assert 'CLs_exp' in d

        results_exp.append(d['CLs_exp'])
        results_obs.append(d['CLs_obs'])

    import numpy as np
    import itertools

    for pair in itertools.combinations(results_exp, r=2):
        assert not np.array_equal(*pair)

    assert len(list(set(results_obs))) == len(pois)

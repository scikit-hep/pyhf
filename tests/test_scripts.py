import json
import shlex
import pyhf
import time
import pytest


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
    pyhf.utils.validate(spec, 'model.json')


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
    command = 'pyhf xml2json validation/xmlimport_input/config/example.xml --basedir validation/xmlimport_input/'
    ret = script_runner.run(*shlex.split(command))
    assert ret.success
    assert ret.stdout != ''
    assert ret.stderr != ''
    d = json.loads(ret.stdout)
    assert d


def test_import_prepHistFactory_and_fit(tmpdir, script_runner):
    temp = tmpdir.join("parsed_output.json")
    command = "pyhf xml2json validation/xmlimport_input/config/example.xml --basedir validation/xmlimport_input/ --output-file {0:s}".format(
        temp.strpath
    )
    ret = script_runner.run(*shlex.split(command))

    command = "pyhf fit {0:s}".format(temp.strpath)
    ret = script_runner.run(*shlex.split(command))

    assert ret.success
    ret_json = json.loads(ret.stdout)
    assert ret_json
    assert "mle_parameters" in ret_json
    assert "twice_nll" not in ret_json

    for measurement in [
        "GaussExample",
        "GammaExample",
        "LogNormExample",
        "ConstExample",
    ]:
        command = "pyhf fit {0:s} --value --measurement {1:s}".format(
            temp.strpath, measurement
        )
        ret = script_runner.run(*shlex.split(command))

        assert ret.success
        ret_json = json.loads(ret.stdout)
        assert ret_json
        assert "mle_parameters" in ret_json
        assert "twice_nll" in ret_json

        tmp_out = tmpdir.join("{0:s}_output.json".format(measurement))
        # make sure output file works too
        command += " --output-file {0:s}".format(tmp_out.strpath)
        ret = script_runner.run(*shlex.split(command))
        assert ret.success
        ret_json = json.load(tmp_out)
        assert "mle_parameters" in ret_json
        assert "twice_nll" in ret_json


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


@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "pytorch", "jax"])
def test_fit_backend_option(tmpdir, script_runner, backend):
    temp = tmpdir.join("parsed_output.json")
    command = "pyhf xml2json validation/xmlimport_input/config/example.xml --basedir validation/xmlimport_input/ --output-file {0:s}".format(
        temp.strpath
    )
    ret = script_runner.run(*shlex.split(command))

    command = "pyhf fit --backend {0:s} {1:s}".format(backend, temp.strpath)
    ret = script_runner.run(*shlex.split(command))

    assert ret.success
    ret_json = json.loads(ret.stdout)
    assert ret_json
    assert "mle_parameters" in ret_json


@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "pytorch", "jax"])
def test_cls_backend_option(tmpdir, script_runner, backend):
    temp = tmpdir.join("parsed_output.json")
    command = 'pyhf xml2json validation/xmlimport_input/config/example.xml --basedir validation/xmlimport_input/ --output-file {0:s}'.format(
        temp.strpath
    )
    ret = script_runner.run(*shlex.split(command))

    command = 'pyhf cls --backend {0:s} {1:s}'.format(backend, temp.strpath)
    ret = script_runner.run(*shlex.split(command))

    assert ret.success
    d = json.loads(ret.stdout)
    assert d
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

    patch.write(
        u'''
[{"op": "replace", "path": "/channels/0/samples/0/data", "value": [5,6]}]
    '''
    )

    temp = tmpdir.join("parsed_output.json")
    command = 'pyhf xml2json validation/xmlimport_input/config/example.xml --basedir validation/xmlimport_input/ --output-file {0:s}'.format(
        temp.strpath
    )
    ret = script_runner.run(*shlex.split(command))

    command = 'pyhf cls {0:s} --patch {1:s}'.format(temp.strpath, patch.strpath)
    ret = script_runner.run(*shlex.split(command))
    assert ret.success

    command = 'pyhf json2xml {0:s} --output-dir {1:s} --patch {2:s}'.format(
        temp.strpath, tmpdir.mkdir('output_1').strpath, patch.strpath
    )
    ret = script_runner.run(*shlex.split(command))
    assert ret.success

    command = f'pyhf cls {temp.strpath:s} --patch -'

    ret = script_runner.run(*shlex.split(command), stdin=patch)
    assert ret.success

    command = f"pyhf json2xml {temp.strpath:s} --output-dir {tmpdir.mkdir('output_2').strpath:s} --patch -"
    ret = script_runner.run(*shlex.split(command), stdin=patch)
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

    command = 'pyhf json2xml {0:s} --output-dir {1:s} --patch {2:s}'.format(
        temp.strpath, tmpdir.mkdir('output').strpath, patch.strpath
    )
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


@pytest.mark.parametrize("optimizer", ["scipy", "minuit"])
@pytest.mark.parametrize(
    "opts,success", [(["maxiter=1000"], True), (["maxiter=1"], False)]
)
def test_fit_optimizer(tmpdir, script_runner, optimizer, opts, success):
    temp = tmpdir.join("parsed_output.json")
    command = "pyhf xml2json validation/xmlimport_input/config/example.xml --basedir validation/xmlimport_input/ --output-file {0:s}".format(
        temp.strpath
    )
    ret = script_runner.run(*shlex.split(command))

    optconf = " ".join(f"--optconf {opt}" for opt in opts)
    command = f"pyhf fit --optimizer {optimizer} {optconf} {temp.strpath}"
    ret = script_runner.run(*shlex.split(command))

    assert ret.success == success


@pytest.mark.parametrize('optimizer', ['scipy', 'minuit'])
@pytest.mark.parametrize(
    'opts,success', [(['maxiter=1000'], True), (['maxiter=1'], False)]
)
def test_cls_optimizer(tmpdir, script_runner, optimizer, opts, success):
    temp = tmpdir.join("parsed_output.json")
    command = 'pyhf xml2json validation/xmlimport_input/config/example.xml --basedir validation/xmlimport_input/ --output-file {0:s}'.format(
        temp.strpath
    )
    ret = script_runner.run(*shlex.split(command))

    optconf = " ".join(f"--optconf {opt}" for opt in opts)
    command = f'pyhf cls {temp.strpath} --optimizer {optimizer} {optconf}'
    ret = script_runner.run(*shlex.split(command))

    assert ret.success == success


def test_inspect(tmpdir, script_runner):
    temp = tmpdir.join("parsed_output.json")
    command = 'pyhf xml2json validation/xmlimport_input/config/example.xml --basedir validation/xmlimport_input/ --output-file {0:s} --hide-progress'.format(
        temp.strpath
    )
    ret = script_runner.run(*shlex.split(command))

    command = 'pyhf inspect {0:s}'.format(temp.strpath)
    ret = script_runner.run(*shlex.split(command))
    assert ret.success


def test_inspect_outfile(tmpdir, script_runner):
    temp = tmpdir.join("parsed_output.json")
    command = 'pyhf xml2json validation/xmlimport_input/config/example.xml --basedir validation/xmlimport_input/ --output-file {0:s} --hide-progress'.format(
        temp.strpath
    )
    ret = script_runner.run(*shlex.split(command))

    tempout = tmpdir.join("inspect_output.json")
    command = 'pyhf inspect {0:s} --output-file {1:s}'.format(
        temp.strpath, tempout.strpath
    )
    ret = script_runner.run(*shlex.split(command))
    assert ret.success

    summary = json.loads(tempout.read())
    assert [
        'channels',
        'measurements',
        'modifiers',
        'parameters',
        'samples',
        'systematics',
    ] == sorted(summary.keys())
    assert len(summary['channels']) == 1
    assert len(summary['measurements']) == 4
    assert len(summary['modifiers']) == 6
    assert len(summary['parameters']) == 6
    assert len(summary['samples']) == 3
    assert len(summary['systematics']) == 6


def test_prune(tmpdir, script_runner):
    temp = tmpdir.join("parsed_output.json")
    command = 'pyhf xml2json validation/xmlimport_input/config/example.xml --basedir validation/xmlimport_input/ --output-file {0:s} --hide-progress'.format(
        temp.strpath
    )
    ret = script_runner.run(*shlex.split(command))

    command = (
        f"pyhf prune -m staterror_channel1 --measurement GammaExample {temp.strpath:s}"
    )
    ret = script_runner.run(*shlex.split(command))
    assert ret.success


def test_prune_outfile(tmpdir, script_runner):
    temp = tmpdir.join("parsed_output.json")
    command = 'pyhf xml2json validation/xmlimport_input/config/example.xml --basedir validation/xmlimport_input/ --output-file {0:s} --hide-progress'.format(
        temp.strpath
    )
    ret = script_runner.run(*shlex.split(command))

    tempout = tmpdir.join("prune_output.json")
    command = 'pyhf prune -m staterror_channel1 --measurement GammaExample {0:s} --output-file {1:s}'.format(
        temp.strpath, tempout.strpath
    )
    ret = script_runner.run(*shlex.split(command))
    assert ret.success

    spec = json.loads(temp.read())
    ws = pyhf.Workspace(spec)
    assert 'GammaExample' in ws.measurement_names
    assert 'staterror_channel1' in ws.parameters
    pruned_spec = json.loads(tempout.read())
    pruned_ws = pyhf.Workspace(pruned_spec)
    assert 'GammaExample' not in pruned_ws.measurement_names
    assert 'staterror_channel1' not in pruned_ws.parameters


def test_rename(tmpdir, script_runner):
    temp = tmpdir.join("parsed_output.json")
    command = 'pyhf xml2json validation/xmlimport_input/config/example.xml --basedir validation/xmlimport_input/ --output-file {0:s} --hide-progress'.format(
        temp.strpath
    )
    ret = script_runner.run(*shlex.split(command))

    command = 'pyhf rename -m staterror_channel1 staterror_channelone --measurement GammaExample GamEx {0:s}'.format(
        temp.strpath
    )
    ret = script_runner.run(*shlex.split(command))
    assert ret.success


def test_rename_outfile(tmpdir, script_runner):
    temp = tmpdir.join("parsed_output.json")
    command = 'pyhf xml2json validation/xmlimport_input/config/example.xml --basedir validation/xmlimport_input/ --output-file {0:s} --hide-progress'.format(
        temp.strpath
    )
    ret = script_runner.run(*shlex.split(command))

    tempout = tmpdir.join("rename_output.json")
    command = 'pyhf rename -m staterror_channel1 staterror_channelone --measurement GammaExample GamEx {0:s} --output-file {1:s}'.format(
        temp.strpath, tempout.strpath
    )
    ret = script_runner.run(*shlex.split(command))
    assert ret.success

    spec = json.loads(temp.read())
    ws = pyhf.Workspace(spec)
    assert 'GammaExample' in ws.measurement_names
    assert 'GamEx' not in ws.measurement_names
    assert 'staterror_channel1' in ws.parameters
    assert 'staterror_channelone' not in ws.parameters
    renamed_spec = json.loads(tempout.read())
    renamed_ws = pyhf.Workspace(renamed_spec)
    assert 'GammaExample' not in renamed_ws.measurement_names
    assert 'GamEx' in renamed_ws.measurement_names
    assert 'staterror_channel1' not in renamed_ws.parameters
    assert 'staterror_channelone' in renamed_ws.parameters


def test_combine(tmpdir, script_runner):
    temp_1 = tmpdir.join("parsed_output.json")
    temp_2 = tmpdir.join("renamed_output.json")
    command = 'pyhf xml2json validation/xmlimport_input/config/example.xml --basedir validation/xmlimport_input/ --output-file {0:s} --hide-progress'.format(
        temp_1.strpath
    )
    ret = script_runner.run(*shlex.split(command))

    rename_channels = {'channel1': 'channel2'}
    rename_measurements = {
        'ConstExample': 'OtherConstExample',
        'LogNormExample': 'OtherLogNormExample',
        'GaussExample': 'OtherGaussExample',
        'GammaExample': 'OtherGammaExample',
    }

    command = 'pyhf rename {0:s} {1:s} {2:s} --output-file {3:s}'.format(
        temp_1.strpath,
        ''.join(' -c ' + ' '.join(item) for item in rename_channels.items()),
        ''.join(
            ' --measurement ' + ' '.join(item) for item in rename_measurements.items()
        ),
        temp_2.strpath,
    )
    ret = script_runner.run(*shlex.split(command))

    command = 'pyhf combine {0:s} {1:s}'.format(temp_1.strpath, temp_2.strpath)
    ret = script_runner.run(*shlex.split(command))
    assert ret.success


def test_combine_outfile(tmpdir, script_runner):
    temp_1 = tmpdir.join("parsed_output.json")
    temp_2 = tmpdir.join("renamed_output.json")
    command = 'pyhf xml2json validation/xmlimport_input/config/example.xml --basedir validation/xmlimport_input/ --output-file {0:s} --hide-progress'.format(
        temp_1.strpath
    )
    ret = script_runner.run(*shlex.split(command))

    rename_channels = {'channel1': 'channel2'}
    rename_measurements = {
        'ConstExample': 'OtherConstExample',
        'LogNormExample': 'OtherLogNormExample',
        'GaussExample': 'OtherGaussExample',
        'GammaExample': 'OtherGammaExample',
    }

    command = 'pyhf rename {0:s} {1:s} {2:s} --output-file {3:s}'.format(
        temp_1.strpath,
        ''.join(' -c ' + ' '.join(item) for item in rename_channels.items()),
        ''.join(
            ' --measurement ' + ' '.join(item) for item in rename_measurements.items()
        ),
        temp_2.strpath,
    )
    ret = script_runner.run(*shlex.split(command))

    tempout = tmpdir.join("combined_output.json")
    command = 'pyhf combine {0:s} {1:s} --output-file {2:s}'.format(
        temp_1.strpath, temp_2.strpath, tempout.strpath
    )
    ret = script_runner.run(*shlex.split(command))
    assert ret.success

    combined_spec = json.loads(tempout.read())
    combined_ws = pyhf.Workspace(combined_spec)
    assert combined_ws.channels == ['channel1', 'channel2']
    assert len(combined_ws.measurement_names) == 8


@pytest.mark.parametrize('do_json', [False, True])
@pytest.mark.parametrize(
    'algorithms', [['md5'], ['sha256'], ['sha256', 'md5'], ['sha256', 'md5']]
)
def test_workspace_digest(tmpdir, script_runner, algorithms, do_json):
    results = {
        'md5': '202eb7615102c35ba86be47eb6fa5e78',
        'sha256': '7c32ca3b8db75cbafcf5cd7ed4672fa2b1fa69e391c9b89068dd947a521866ec',
    }

    temp = tmpdir.join("parsed_output.json")
    command = f'pyhf xml2json validation/xmlimport_input/config/example.xml --basedir validation/xmlimport_input/ --output-file {temp.strpath} --hide-progress'
    ret = script_runner.run(*shlex.split(command))

    command = f"pyhf digest {temp.strpath} -a {' -a '.join(algorithms)}{' -j' if do_json else ''}"
    ret = script_runner.run(*shlex.split(command))
    assert ret.success
    assert all(algorithm in ret.stdout for algorithm in algorithms)
    if do_json:
        expected_output = json.dumps(
            {algorithm: results[algorithm] for algorithm in algorithms},
            sort_keys=True,
            indent=4,
        )
    else:
        expected_output = '\n'.join(
            f"{algorithm}:{results[algorithm]}" for algorithm in algorithms
        )

    assert ret.stdout == expected_output + '\n'
    assert ret.stderr == ''

    if do_json:
        assert json.loads(ret.stdout) == {
            algorithm: results[algorithm] for algorithm in algorithms
        }


@pytest.mark.parametrize('output_file', [False, True])
@pytest.mark.parametrize('with_metadata', [False, True])
def test_patchset_extract(datadir, tmpdir, script_runner, output_file, with_metadata):
    temp = tmpdir.join("extracted_output.json")
    command = f'pyhf patchset extract {datadir.join("example_patchset.json").strpath} --name patch_channel1_signal_syst1'
    if output_file:
        command += f" --output-file {temp.strpath}"
    if with_metadata:
        command += " --with-metadata"

    ret = script_runner.run(*shlex.split(command))

    assert ret.success
    if output_file:
        extracted_output = json.loads(temp.read())
    else:
        extracted_output = json.loads(ret.stdout)
    if with_metadata:
        assert 'metadata' in extracted_output
    else:
        assert (
            extracted_output
            == json.load(datadir.join("example_patchset.json"))['patches'][0]['patch']
        )


def test_patchset_verify(datadir, script_runner):
    command = f'pyhf patchset verify {datadir.join("example_bkgonly.json").strpath} {datadir.join("example_patchset.json").strpath}'
    ret = script_runner.run(*shlex.split(command))

    assert ret.success
    assert 'All good' in ret.stdout


@pytest.mark.parametrize('output_file', [False, True])
def test_patchset_apply(datadir, tmpdir, script_runner, output_file):
    temp = tmpdir.join("patched_output.json")
    command = f'pyhf patchset apply {datadir.join("example_bkgonly.json").strpath} {datadir.join("example_patchset.json").strpath} --name patch_channel1_signal_syst1'
    if output_file:
        command += f" --output-file {temp.strpath}"

    ret = script_runner.run(*shlex.split(command))

    assert ret.success
    if output_file:
        extracted_output = json.loads(temp.read())
    else:
        extracted_output = json.loads(ret.stdout)
    assert extracted_output['channels'][0]['samples'][0]['modifiers'][0]['data'] == {
        "hi": 1.2,
        "lo": 0.8,
    }


def test_sort(tmpdir, script_runner):
    temp = tmpdir.join("parsed_output.json")
    command = 'pyhf xml2json validation/xmlimport_input/config/example.xml --basedir validation/xmlimport_input/ --output-file {0:s} --hide-progress'.format(
        temp.strpath
    )
    ret = script_runner.run(*shlex.split(command))

    command = f'pyhf sort {temp.strpath}'

    ret = script_runner.run(*shlex.split(command))
    assert ret.success


def test_sort_outfile(tmpdir, script_runner):
    temp = tmpdir.join("parsed_output.json")
    command = 'pyhf xml2json validation/xmlimport_input/config/example.xml --basedir validation/xmlimport_input/ --output-file {0:s} --hide-progress'.format(
        temp.strpath
    )
    ret = script_runner.run(*shlex.split(command))

    tempout = tmpdir.join("sort_output.json")
    command = f'pyhf sort {temp.strpath} --output-file {tempout.strpath}'

    ret = script_runner.run(*shlex.split(command))
    assert ret.success

import json
import logging
import shlex
import sys
import tarfile
import time
from importlib import import_module, reload
from unittest import mock

import pytest
from click.testing import CliRunner

import pyhf


@pytest.fixture(scope="function")
def tarfile_path(tmp_path):
    with open(tmp_path.joinpath("test_file.txt"), "w", encoding="utf-8") as write_file:
        write_file.write("test file")
    with tarfile.open(
        tmp_path.joinpath("test_tar.tar.gz"), mode="w:gz", encoding="utf-8"
    ) as archive:
        archive.add(tmp_path.joinpath("test_file.txt"))
    return tmp_path.joinpath("test_tar.tar.gz")


def test_version(script_runner):
    command = 'pyhf --version'
    start = time.time()
    ret = script_runner.run(shlex.split(command))
    end = time.time()
    elapsed = end - start
    assert ret.success
    assert pyhf.__version__ in ret.stdout
    assert ret.stderr == ''
    # make sure it took less than a second
    assert elapsed < 1.0


@pytest.mark.parametrize("flag", ["--cite", "--citation"])
def test_citation(script_runner, flag):
    command = f'pyhf {flag}'
    start = time.time()
    ret = script_runner.run(shlex.split(command))
    end = time.time()
    elapsed = end - start
    assert ret.success
    assert ret.stdout.startswith('@software{pyhf,')
    assert '@article{pyhf_joss,' in ret.stdout
    # ensure there's not \n\n at the end
    assert ret.stdout.endswith('}\n')
    # make sure it took less than a second
    assert elapsed < 1.0


# see test_import.py for the same (detailed) test
def test_import_prepHistFactory(tmp_path, script_runner):
    temp = tmp_path.joinpath("parsed_output.json")
    command = f'pyhf xml2json validation/xmlimport_input/config/example.xml --basedir validation/xmlimport_input/ --output-file {temp} --hide-progress'
    ret = script_runner.run(shlex.split(command))
    assert ret.success
    assert ret.stdout == ''
    assert ret.stderr == ''

    parsed_xml = json.loads(temp.read_text())
    spec = {'channels': parsed_xml['channels']}
    pyhf.schema.validate(spec, 'model.json')


def test_import_prepHistFactory_withProgress(tmp_path, script_runner):
    temp = tmp_path.joinpath("parsed_output.json")
    command = f'pyhf xml2json validation/xmlimport_input/config/example.xml --basedir validation/xmlimport_input/ --output-file {temp}'
    ret = script_runner.run(shlex.split(command))
    assert ret.success
    assert ret.stdout == ''
    assert ret.stderr != ''


def test_import_prepHistFactory_stdout(tmp_path, script_runner):
    command = 'pyhf xml2json validation/xmlimport_input/config/example.xml --basedir validation/xmlimport_input/'
    ret = script_runner.run(shlex.split(command))
    assert ret.success
    assert ret.stdout != ''
    assert ret.stderr != ''
    d = json.loads(ret.stdout)
    assert d


def test_import_prepHistFactory_and_fit(tmp_path, script_runner):
    temp = tmp_path.joinpath("parsed_output.json")
    command = f"pyhf xml2json validation/xmlimport_input/config/example.xml --basedir validation/xmlimport_input/ --output-file {temp}"
    ret = script_runner.run(shlex.split(command))

    command = f"pyhf fit {temp}"
    ret = script_runner.run(shlex.split(command))

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
        command = f"pyhf fit {temp} --value --measurement {measurement:s}"
        ret = script_runner.run(shlex.split(command))

        assert ret.success
        ret_json = json.loads(ret.stdout)
        assert ret_json
        assert "mle_parameters" in ret_json
        assert "twice_nll" in ret_json

        tmp_out = tmp_path.joinpath(f"{measurement:s}_output.json")
        # make sure output file works too
        command += f" --output-file {tmp_out}"
        ret = script_runner.run(shlex.split(command))
        assert ret.success
        ret_json = json.load(tmp_out.open())
        assert "mle_parameters" in ret_json
        assert "twice_nll" in ret_json


def test_import_prepHistFactory_and_cls(tmp_path, script_runner):
    temp = tmp_path.joinpath("parsed_output.json")
    command = f'pyhf xml2json validation/xmlimport_input/config/example.xml --basedir validation/xmlimport_input/ --output-file {temp}'
    ret = script_runner.run(shlex.split(command))

    command = f'pyhf cls {temp}'
    ret = script_runner.run(shlex.split(command))

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
        command = f'pyhf cls {temp} --measurement {measurement:s}'
        ret = script_runner.run(shlex.split(command))

        assert ret.success
        d = json.loads(ret.stdout)
        assert d
        assert 'CLs_obs' in d
        assert 'CLs_exp' in d

        tmp_out = tmp_path.joinpath(f'{measurement:s}_output.json')
        # make sure output file works too
        command += f' --output-file {tmp_out}'
        ret = script_runner.run(shlex.split(command))
        assert ret.success
        d = json.load(tmp_out.open())
        assert 'CLs_obs' in d
        assert 'CLs_exp' in d


def test_import_usingMounts(datadir, tmp_path, script_runner):
    data = datadir.joinpath("xmlimport_absolutePaths")

    temp = tmp_path.joinpath("parsed_output.json")
    command = f'pyhf xml2json --hide-progress -v {data}:/absolute/path/to -v {data}:/another/absolute/path/to --output-file {temp} {data.joinpath("config/example.xml")}'

    ret = script_runner.run(shlex.split(command))
    assert ret.success
    assert ret.stdout == ''
    assert ret.stderr == ''

    parsed_xml = json.loads(temp.read_text())
    spec = {'channels': parsed_xml['channels']}
    pyhf.schema.validate(spec, 'model.json')


def test_import_usingMounts_badDelimitedPaths(datadir, tmp_path, script_runner):
    data = datadir.joinpath("xmlimport_absolutePaths")

    temp = tmp_path.joinpath("parsed_output.json")
    command = f'pyhf xml2json --hide-progress -v {data}::/absolute/path/to -v {data}/another/absolute/path/to --output-file {temp} {data.joinpath("config/example.xml")}'

    ret = script_runner.run(shlex.split(command))
    assert not ret.success
    assert ret.stdout == ''
    assert 'is not a valid colon-separated option' in ret.stderr


@pytest.mark.parametrize("backend", ["numpy", "pytorch", "jax"])
def test_fit_backend_option(tmp_path, script_runner, backend):
    temp = tmp_path.joinpath("parsed_output.json")
    command = f"pyhf xml2json validation/xmlimport_input/config/example.xml --basedir validation/xmlimport_input/ --output-file {temp}"
    ret = script_runner.run(shlex.split(command))

    command = f"pyhf fit --backend {backend:s} {temp}"
    ret = script_runner.run(shlex.split(command))

    assert ret.success
    ret_json = json.loads(ret.stdout)
    assert ret_json
    assert "mle_parameters" in ret_json


@pytest.mark.parametrize("backend", ["numpy", "pytorch", "jax"])
def test_cls_backend_option(tmp_path, script_runner, backend):
    temp = tmp_path.joinpath("parsed_output.json")
    command = f'pyhf xml2json validation/xmlimport_input/config/example.xml --basedir validation/xmlimport_input/ --output-file {temp}'
    ret = script_runner.run(shlex.split(command))

    command = f'pyhf cls --backend {backend:s} {temp}'
    ret = script_runner.run(shlex.split(command))

    assert ret.success
    d = json.loads(ret.stdout)
    assert d
    assert 'CLs_obs' in d
    assert 'CLs_exp' in d


def test_import_and_export(tmp_path, script_runner):
    temp = tmp_path.joinpath("parsed_output.json")
    command = f'pyhf xml2json validation/xmlimport_input/config/example.xml --basedir validation/xmlimport_input/ --output-file {temp}'
    ret = script_runner.run(shlex.split(command))

    output_dir_path = tmp_path / "output"
    output_dir_path.mkdir()

    command = f"pyhf json2xml {temp} --output-dir {output_dir_path}"
    ret = script_runner.run(shlex.split(command))
    assert ret.success


def test_patch(tmp_path, script_runner):
    patch = tmp_path.joinpath('patch.json')

    patch.write_text(
        '''
[{"op": "replace", "path": "/channels/0/samples/0/data", "value": [5,6]}]
    '''
    )

    temp = tmp_path.joinpath("parsed_output.json")
    command = f'pyhf xml2json validation/xmlimport_input/config/example.xml --basedir validation/xmlimport_input/ --output-file {temp}'
    ret = script_runner.run(shlex.split(command))

    command = f'pyhf cls {temp} --patch {patch}'
    ret = script_runner.run(shlex.split(command))
    assert ret.success

    output_dir_path = tmp_path / "output_1"
    output_dir_path.mkdir(exist_ok=True)

    command = f"pyhf json2xml {temp} --output-dir {output_dir_path} --patch {patch}"
    ret = script_runner.run(shlex.split(command))
    assert ret.success

    command = f'pyhf cls {temp} --patch -'

    ret = script_runner.run(shlex.split(command), stdin=patch.open())
    assert ret.success

    output_dir_path = tmp_path / "output_2"
    output_dir_path.mkdir(exist_ok=True)

    command = f"pyhf json2xml {temp} --output-dir {output_dir_path} --patch -"
    ret = script_runner.run(shlex.split(command), stdin=patch.open())
    assert ret.success


def test_patch_fail(tmp_path, script_runner):
    patch = tmp_path.joinpath('patch.json')

    patch.write_text('''not,json''')

    temp = tmp_path.joinpath("parsed_output.json")
    command = f'pyhf xml2json validation/xmlimport_input/config/example.xml --basedir validation/xmlimport_input/ --output-file {temp}'
    ret = script_runner.run(shlex.split(command))

    command = f'pyhf cls {temp} --patch {patch}'
    ret = script_runner.run(shlex.split(command))
    assert not ret.success

    output_dir_path = tmp_path / "output"
    output_dir_path.mkdir()

    command = f"pyhf json2xml {temp} --output-dir {output_dir_path} --patch {patch}"
    ret = script_runner.run(shlex.split(command))
    assert not ret.success


def test_bad_measurement_name(tmp_path, script_runner):
    temp = tmp_path.joinpath("parsed_output.json")
    command = f'pyhf xml2json validation/xmlimport_input/config/example.xml --basedir validation/xmlimport_input/ --output-file {temp}'
    ret = script_runner.run(shlex.split(command))

    command = f'pyhf cls {temp} --measurement "a-fake-measurement-name"'
    ret = script_runner.run(shlex.split(command))
    assert not ret.success
    # assert 'no measurement by name' in ret.stderr  # numpy swallows the log.error() here, dunno why


def test_testpoi(tmp_path, script_runner):
    temp = tmp_path.joinpath("parsed_output.json")
    command = f'pyhf xml2json validation/xmlimport_input/config/example.xml --basedir validation/xmlimport_input/ --output-file {temp}'
    ret = script_runner.run(shlex.split(command))

    pois = [1.0, 0.5, 0.001]
    results_exp = []
    results_obs = []
    for test_poi in pois:
        command = f'pyhf cls {temp} --test-poi {test_poi:f}'
        ret = script_runner.run(shlex.split(command))

        assert ret.success
        d = json.loads(ret.stdout)
        assert d
        assert 'CLs_obs' in d
        assert 'CLs_exp' in d

        results_exp.append(d['CLs_exp'])
        results_obs.append(d['CLs_obs'])

    import itertools

    import numpy as np

    for pair in itertools.combinations(results_exp, r=2):
        assert not np.array_equal(*pair)

    assert len(list(set(results_obs))) == len(pois)


@pytest.mark.parametrize("optimizer", ["scipy", "minuit"])
@pytest.mark.parametrize(
    "opts,success", [(["maxiter=1000"], True), (["maxiter=1"], False)]
)
def test_fit_optimizer(tmp_path, script_runner, optimizer, opts, success):
    temp = tmp_path.joinpath("parsed_output.json")
    command = f"pyhf xml2json validation/xmlimport_input/config/example.xml --basedir validation/xmlimport_input/ --output-file {temp}"
    ret = script_runner.run(shlex.split(command))

    optconf = " ".join(f"--optconf {opt}" for opt in opts)
    command = f"pyhf fit --optimizer {optimizer} {optconf} {temp}"
    ret = script_runner.run(shlex.split(command))

    assert ret.success == success


@pytest.mark.parametrize('optimizer', ['scipy', 'minuit'])
@pytest.mark.parametrize(
    'opts,success', [(['maxiter=1000'], True), (['maxiter=1'], False)]
)
def test_cls_optimizer(tmp_path, script_runner, optimizer, opts, success):
    temp = tmp_path.joinpath("parsed_output.json")
    command = f'pyhf xml2json validation/xmlimport_input/config/example.xml --basedir validation/xmlimport_input/ --output-file {temp}'
    ret = script_runner.run(shlex.split(command))

    optconf = " ".join(f"--optconf {opt}" for opt in opts)
    command = f'pyhf cls {temp} --optimizer {optimizer} {optconf}'
    ret = script_runner.run(shlex.split(command))

    assert ret.success == success


def test_inspect(tmp_path, script_runner):
    temp = tmp_path.joinpath("parsed_output.json")
    command = f'pyhf xml2json validation/xmlimport_input/config/example.xml --basedir validation/xmlimport_input/ --output-file {temp} --hide-progress'
    ret = script_runner.run(shlex.split(command))

    command = f'pyhf inspect {temp}'
    ret = script_runner.run(shlex.split(command))
    assert ret.success


def test_inspect_outfile(tmp_path, script_runner):
    temp = tmp_path.joinpath("parsed_output.json")
    command = f'pyhf xml2json validation/xmlimport_input/config/example.xml --basedir validation/xmlimport_input/ --output-file {temp} --hide-progress'
    ret = script_runner.run(shlex.split(command))

    tempout = tmp_path.joinpath("inspect_output.json")
    command = f'pyhf inspect {temp} --output-file {tempout}'
    ret = script_runner.run(shlex.split(command))
    assert ret.success

    summary = json.loads(tempout.read_text())
    assert [
        'channels',
        'measurements',
        'modifiers',
        'parameters',
        'samples',
        'systematics',
    ] == sorted(summary)
    assert len(summary['channels']) == 1
    assert len(summary['measurements']) == 4
    assert len(summary['modifiers']) == 6
    assert len(summary['parameters']) == 6
    assert len(summary['samples']) == 3
    assert len(summary['systematics']) == 6


def test_prune(tmp_path, script_runner):
    temp = tmp_path.joinpath("parsed_output.json")
    command = f'pyhf xml2json validation/xmlimport_input/config/example.xml --basedir validation/xmlimport_input/ --output-file {temp} --hide-progress'
    ret = script_runner.run(shlex.split(command))

    command = f"pyhf prune -m staterror_channel1 --measurement GammaExample {temp}"
    ret = script_runner.run(shlex.split(command))
    assert ret.success


def test_prune_outfile(tmp_path, script_runner):
    temp = tmp_path.joinpath("parsed_output.json")
    command = f'pyhf xml2json validation/xmlimport_input/config/example.xml --basedir validation/xmlimport_input/ --output-file {temp} --hide-progress'
    ret = script_runner.run(shlex.split(command))

    tempout = tmp_path.joinpath("prune_output.json")
    command = f'pyhf prune -m staterror_channel1 --measurement GammaExample {temp} --output-file {tempout}'
    ret = script_runner.run(shlex.split(command))
    assert ret.success

    spec = json.loads(temp.read_text())
    ws = pyhf.Workspace(spec)
    assert 'GammaExample' in ws.measurement_names
    assert 'staterror_channel1' in ws.model().config.parameters
    pruned_spec = json.loads(tempout.read_text())
    pruned_ws = pyhf.Workspace(pruned_spec)
    assert 'GammaExample' not in pruned_ws.measurement_names
    assert 'staterror_channel1' not in pruned_ws.model().config.parameters


def test_rename(tmp_path, script_runner):
    temp = tmp_path.joinpath("parsed_output.json")
    command = f'pyhf xml2json validation/xmlimport_input/config/example.xml --basedir validation/xmlimport_input/ --output-file {temp} --hide-progress'
    ret = script_runner.run(shlex.split(command))

    command = f'pyhf rename -m staterror_channel1 staterror_channelone --measurement GammaExample GamEx {temp}'
    ret = script_runner.run(shlex.split(command))
    assert ret.success


def test_rename_outfile(tmp_path, script_runner):
    temp = tmp_path.joinpath("parsed_output.json")
    command = f'pyhf xml2json validation/xmlimport_input/config/example.xml --basedir validation/xmlimport_input/ --output-file {temp} --hide-progress'
    ret = script_runner.run(shlex.split(command))

    tempout = tmp_path.joinpath("rename_output.json")
    command = f'pyhf rename -m staterror_channel1 staterror_channelone --measurement GammaExample GamEx {temp} --output-file {tempout}'
    ret = script_runner.run(shlex.split(command))
    assert ret.success

    spec = json.loads(temp.read_text())
    ws = pyhf.Workspace(spec)
    assert 'GammaExample' in ws.measurement_names
    assert 'GamEx' not in ws.measurement_names
    assert 'staterror_channel1' in ws.model().config.parameters
    assert 'staterror_channelone' not in ws.model().config.parameters
    renamed_spec = json.loads(tempout.read_text())
    renamed_ws = pyhf.Workspace(renamed_spec)
    assert 'GammaExample' not in renamed_ws.measurement_names
    assert 'GamEx' in renamed_ws.measurement_names
    assert 'staterror_channel1' not in renamed_ws.model().config.parameters
    assert 'staterror_channelone' in renamed_ws.model().config.parameters


def test_combine(tmp_path, script_runner):
    temp_1 = tmp_path.joinpath("parsed_output.json")
    temp_2 = tmp_path.joinpath("renamed_output.json")
    command = f'pyhf xml2json validation/xmlimport_input/config/example.xml --basedir validation/xmlimport_input/ --output-file {temp_1} --hide-progress'
    ret = script_runner.run(shlex.split(command))

    rename_channels = {'channel1': 'channel2'}
    rename_measurements = {
        'ConstExample': 'OtherConstExample',
        'LogNormExample': 'OtherLogNormExample',
        'GaussExample': 'OtherGaussExample',
        'GammaExample': 'OtherGammaExample',
    }

    _opts_channels = ''.join(
        ' -c ' + ' '.join(item) for item in rename_channels.items()
    )
    _opts_measurements = ''.join(
        ' --measurement ' + ' '.join(item) for item in rename_measurements.items()
    )
    command = f"pyhf rename {temp_1} {_opts_channels:s} {_opts_measurements:s} --output-file {temp_2}"
    ret = script_runner.run(shlex.split(command))

    command = f'pyhf combine {temp_1} {temp_2}'
    ret = script_runner.run(shlex.split(command))
    assert ret.success


def test_combine_outfile(tmp_path, script_runner):
    temp_1 = tmp_path.joinpath("parsed_output.json")
    temp_2 = tmp_path.joinpath("renamed_output.json")
    command = f'pyhf xml2json validation/xmlimport_input/config/example.xml --basedir validation/xmlimport_input/ --output-file {temp_1} --hide-progress'
    ret = script_runner.run(shlex.split(command))

    rename_channels = {'channel1': 'channel2'}
    rename_measurements = {
        'ConstExample': 'OtherConstExample',
        'LogNormExample': 'OtherLogNormExample',
        'GaussExample': 'OtherGaussExample',
        'GammaExample': 'OtherGammaExample',
    }

    _opts_channels = ''.join(
        ' -c ' + ' '.join(item) for item in rename_channels.items()
    )
    _opts_measurements = ''.join(
        ' --measurement ' + ' '.join(item) for item in rename_measurements.items()
    )
    command = f"pyhf rename {temp_1} {_opts_channels:s} {_opts_measurements:s} --output-file {temp_2}"
    ret = script_runner.run(shlex.split(command))

    tempout = tmp_path.joinpath("combined_output.json")
    command = f'pyhf combine {temp_1} {temp_2} --output-file {tempout}'
    ret = script_runner.run(shlex.split(command))
    assert ret.success

    combined_spec = json.loads(tempout.read_text())
    combined_ws = pyhf.Workspace(combined_spec)
    assert combined_ws.channels == ['channel1', 'channel2']
    assert len(combined_ws.measurement_names) == 8


def test_combine_merge_channels(tmp_path, script_runner):
    temp_1 = tmp_path.joinpath("parsed_output.json")
    temp_2 = tmp_path.joinpath("renamed_output.json")
    command = f'pyhf xml2json validation/xmlimport_input/config/example.xml --basedir validation/xmlimport_input/ --output-file {temp_1} --hide-progress'
    ret = script_runner.run(shlex.split(command))
    assert ret.success

    command = f'pyhf prune {temp_1} --sample signal --output-file {temp_2}'

    ret = script_runner.run(shlex.split(command))
    assert ret.success

    command = f'pyhf combine --merge-channels --join "left outer" {temp_1} {temp_2}'
    ret = script_runner.run(shlex.split(command))
    assert ret.success


@pytest.mark.parametrize('do_json', [False, True])
@pytest.mark.parametrize(
    'algorithms', [['md5'], ['sha256'], ['sha256', 'md5'], ['sha256', 'md5']]
)
def test_workspace_digest(tmp_path, script_runner, algorithms, do_json):
    results = {
        'md5': '7de8930ff37e5a4f6a31da11bda7813f',
        'sha256': '6d416ee67a40460499ea2ef596fb1e682a563d7df06e690018a211d35238aecc',
    }

    temp = tmp_path.joinpath("parsed_output.json")
    command = f'pyhf xml2json validation/xmlimport_input/config/example.xml --basedir validation/xmlimport_input/ --output-file {temp} --hide-progress'
    ret = script_runner.run(shlex.split(command))

    command = (
        f"pyhf digest {temp} -a {' -a '.join(algorithms)}{' -j' if do_json else ''}"
    )
    ret = script_runner.run(shlex.split(command))
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


@pytest.mark.parametrize(
    "archive",
    [
        "https://www.hepdata.net/record/resource/1408476?view=true",
        "https://doi.org/10.17182/hepdata.89408.v1/r2",
    ],
)
def test_patchset_download(
    tmp_path, script_runner, requests_mock, tarfile_path, archive
):
    requests_mock.get(archive, content=open(tarfile_path, "rb").read())
    command = f'pyhf contrib download {archive} {tmp_path.joinpath("likelihoods")}'
    ret = script_runner.run(shlex.split(command))
    assert ret.success

    # Run with all optional flags
    command = f'pyhf contrib download --verbose --force {archive} {tmp_path.joinpath("likelihoods")}'
    ret = script_runner.run(shlex.split(command))
    assert ret.success

    requests_mock.get(
        "https://www.pyhfthisdoesnotexist.org/record/resource/1234567", status_code=200
    )
    command = f'pyhf contrib download --verbose https://www.pyhfthisdoesnotexist.org/record/resource/1234567 {tmp_path.joinpath("likelihoods")}'
    ret = script_runner.run(shlex.split(command))
    assert not ret.success
    assert (
        "pyhf.exceptions.InvalidArchiveHost: www.pyhfthisdoesnotexist.org is not an approved archive host"
        in ret.stderr
    )

    # httpstat.us is a real website that can be used for testing responses
    requests_mock.get(
        "https://httpstat.us/404/record/resource/1234567", status_code=404
    )
    command = f'pyhf contrib download --verbose --force https://httpstat.us/404/record/resource/1234567 {tmp_path.joinpath("likelihoods")}'
    ret = script_runner.run(shlex.split(command))
    assert not ret.success
    assert "gives a response code of 404" in ret.stderr


def test_missing_contrib_extra(caplog):
    with mock.patch.dict(sys.modules):
        sys.modules["requests"] = None
        if "pyhf.contrib.utils" in sys.modules:
            reload(sys.modules["pyhf.contrib.utils"])
        else:
            import_module("pyhf.contrib.utils")

    with caplog.at_level(logging.ERROR):
        for line in [
            "import of requests halted; None in sys.modules",
            "Installation of the contrib extra is required to use pyhf.contrib.utils.download",
            "Please install with: python -m pip install 'pyhf[contrib]'",
        ]:
            assert line in caplog.text
        caplog.clear()


def test_missing_contrib_download(caplog):
    with mock.patch.dict(sys.modules):
        sys.modules["requests"] = None
        if "pyhf.contrib.utils" in sys.modules:
            reload(sys.modules["pyhf.contrib.utils"])
        else:
            import_module("pyhf.contrib.utils")

        # Force environment for runner
        for module in [
            "pyhf.contrib",
            "pyhf.contrib.cli",
            "pyhf.contrib.utils",
        ]:
            if module in sys.modules:
                del sys.modules[module]

        from pyhf.contrib.cli import download

        # mix_stderr removed in Click v8.2.0.
        # Can simplify once pyhf is Python 3.10+.
        try:
            runner = CliRunner(mix_stderr=False)
        except TypeError:
            runner = CliRunner()
        result = runner.invoke(
            download,
            [
                "--verbose",
                "https://www.hepdata.net/record/resource/1408476?view=true",
                "1Lbb-likelihoods",
            ],
        )
        assert result.exit_code == 0

        with caplog.at_level(logging.ERROR):
            for line in [
                "module 'pyhf.contrib.utils' has no attribute 'download'",
                "Installation of the contrib extra is required to use the contrib CLI API",
                "Please install with: python -m pip install 'pyhf[contrib]'",
            ]:
                assert line in caplog.text
            caplog.clear()


def test_patchset_inspect(datadir, script_runner):
    command = f'pyhf patchset inspect {datadir.joinpath("example_patchset.json")}'
    ret = script_runner.run(shlex.split(command))
    assert 'patch_channel1_signal_syst1' in ret.stdout


@pytest.mark.parametrize('output_file', [False, True])
@pytest.mark.parametrize('with_metadata', [False, True])
def test_patchset_extract(datadir, tmp_path, script_runner, output_file, with_metadata):
    temp = tmp_path.joinpath("extracted_output.json")
    command = f'pyhf patchset extract {datadir.joinpath("example_patchset.json")} --name patch_channel1_signal_syst1'
    if output_file:
        command += f" --output-file {temp}"
    if with_metadata:
        command += " --with-metadata"

    ret = script_runner.run(shlex.split(command))

    assert ret.success
    if output_file:
        extracted_output = json.loads(temp.read_text())
    else:
        extracted_output = json.loads(ret.stdout)
    if with_metadata:
        assert 'metadata' in extracted_output
    else:
        assert (
            extracted_output
            == json.load(
                datadir.joinpath("example_patchset.json").open(encoding="utf-8")
            )["patches"][0]["patch"]
        )


def test_patchset_verify(datadir, script_runner):
    command = f'pyhf patchset verify {datadir.joinpath("example_bkgonly.json")} {datadir.joinpath("example_patchset.json")}'
    ret = script_runner.run(shlex.split(command))

    assert ret.success
    assert 'All good' in ret.stdout


@pytest.mark.parametrize('output_file', [False, True])
def test_patchset_apply(datadir, tmp_path, script_runner, output_file):
    temp = tmp_path.joinpath("patched_output.json")
    command = f'pyhf patchset apply {datadir.joinpath("example_bkgonly.json")} {datadir.joinpath("example_patchset.json")} --name patch_channel1_signal_syst1'
    if output_file:
        command += f" --output-file {temp}"

    ret = script_runner.run(shlex.split(command))

    assert ret.success
    if output_file:
        extracted_output = json.loads(temp.read_text())
    else:
        extracted_output = json.loads(ret.stdout)
    assert extracted_output['channels'][0]['samples'][0]['modifiers'][0]['data'] == {
        "hi": 1.2,
        "lo": 0.8,
    }


def test_sort(tmp_path, script_runner):
    temp = tmp_path.joinpath("parsed_output.json")
    command = f'pyhf xml2json validation/xmlimport_input/config/example.xml --basedir validation/xmlimport_input/ --output-file {temp} --hide-progress'
    ret = script_runner.run(shlex.split(command))

    command = f'pyhf sort {temp}'

    ret = script_runner.run(shlex.split(command))
    assert ret.success


def test_sort_outfile(tmp_path, script_runner):
    temp = tmp_path.joinpath("parsed_output.json")
    command = f'pyhf xml2json validation/xmlimport_input/config/example.xml --basedir validation/xmlimport_input/ --output-file {temp} --hide-progress'
    ret = script_runner.run(shlex.split(command))

    tempout = tmp_path.joinpath("sort_output.json")
    command = f'pyhf sort {temp} --output-file {tempout}'

    ret = script_runner.run(shlex.split(command))
    assert ret.success

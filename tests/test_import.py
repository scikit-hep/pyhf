import pyhf
import pyhf.readxml
import numpy as np
import uproot
from pathlib import Path
import pytest
import xml.etree.ElementTree as ET
import logging
from jsonschema import ValidationError


def assert_equal_dictionary(d1, d2):
    "recursively compare 2 dictionaries"
    for k in d1:
        assert k in d2
        if isinstance(d1[k], dict):
            assert_equal_dictionary(d1[k], d2[k])
        else:
            assert d1[k] == d2[k]


def test_dedupe_parameters():
    parameters = [
        {'name': 'SigXsecOverSM', 'bounds': [[0.0, 10.0]]},
        {'name': 'SigXsecOverSM', 'bounds': [[0.0, 10.0]]},
    ]
    assert len(pyhf.readxml.dedupe_parameters(parameters)) == 1
    parameters[1]['bounds'] = [[0.0, 2.0]]
    with pytest.raises(RuntimeError, match="SigXsecOverSM"):
        pyhf.readxml.dedupe_parameters(parameters)


def test_process_normfactor_configs():
    # Check to see if mu_ttbar NormFactor is overridden correctly
    # - ParamSetting has a config for it
    # - other_parameter_configs has a config for it
    # Make sure that when two measurements exist, we're copying things across correctly
    toplvl = ET.Element("Combination")
    meas = ET.Element(
        "Measurement",
        Name='NormalMeasurement',
        Lumi=str(1.0),
        LumiRelErr=str(0.017),
        ExportOnly=str(True),
    )
    poiel = ET.Element('POI')
    poiel.text = 'mu_SIG'
    meas.append(poiel)

    setting = ET.Element('ParamSetting', Const='True')
    setting.text = ' '.join(['Lumi', 'alpha_mu_both', 'alpha_mu_paramSettingOnly'])
    meas.append(setting)

    setting = ET.Element('ParamSetting', Val='2.0')
    setting.text = ' '.join(['alpha_mu_both'])
    meas.append(setting)

    toplvl.append(meas)

    meas = ET.Element(
        "Measurement",
        Name='ParallelMeasurement',
        Lumi=str(1.0),
        LumiRelErr=str(0.017),
        ExportOnly=str(True),
    )
    poiel = ET.Element('POI')
    poiel.text = 'mu_BKG'
    meas.append(poiel)

    setting = ET.Element('ParamSetting', Val='3.0')
    setting.text = ' '.join(['alpha_mu_both'])
    meas.append(setting)

    toplvl.append(meas)

    other_parameter_configs = [
        dict(name='mu_both', inits=[1.0], bounds=[[1.0, 5.0]], fixed=False),
        dict(name='mu_otherConfigOnly', inits=[1.0], bounds=[[0.0, 10.0]], fixed=False),
    ]

    result = pyhf.readxml.process_measurements(
        toplvl, other_parameter_configs=other_parameter_configs
    )
    result = {
        m['name']: {k['name']: k for k in m['config']['parameters']} for m in result
    }
    assert result

    # make sure ParamSetting configs override NormFactor configs
    assert result['NormalMeasurement']['mu_both']['fixed']
    assert result['NormalMeasurement']['mu_both']['inits'] == [2.0]
    assert result['NormalMeasurement']['mu_both']['bounds'] == [[1.0, 5.0]]

    # make sure ParamSetting is doing the right thing
    assert result['NormalMeasurement']['mu_paramSettingOnly']['fixed']
    assert 'inits' not in result['NormalMeasurement']['mu_paramSettingOnly']
    assert 'bounds' not in result['NormalMeasurement']['mu_paramSettingOnly']

    # make sure our code doesn't accidentally override other parameter configs
    assert not result['NormalMeasurement']['mu_otherConfigOnly']['fixed']
    assert result['NormalMeasurement']['mu_otherConfigOnly']['inits'] == [1.0]
    assert result['NormalMeasurement']['mu_otherConfigOnly']['bounds'] == [[0.0, 10.0]]

    # make sure settings from one measurement don't leak to other
    assert not result['ParallelMeasurement']['mu_both']['fixed']
    assert result['ParallelMeasurement']['mu_both']['inits'] == [3.0]
    assert result['ParallelMeasurement']['mu_both']['bounds'] == [[1.0, 5.0]]


def test_import_histogram():
    data, uncert = pyhf.readxml.import_root_histogram(
        lambda x: Path("validation/xmlimport_input/data").joinpath(x),
        "example.root",
        "",
        "data",
    )
    assert data == [122.0, 112.0]
    assert uncert == [11.045360565185547, 10.58300495147705]


def test_import_histogram_KeyError():
    with pytest.raises(KeyError):
        pyhf.readxml.import_root_histogram(
            lambda x: Path("validation/xmlimport_input/data").joinpath(x),
            "example.root",
            "",
            "invalid_key",
        )


def test_import_measurements():
    parsed_xml = pyhf.readxml.parse(
        'validation/xmlimport_input/config/example.xml', 'validation/xmlimport_input/'
    )
    measurements = parsed_xml['measurements']
    assert len(measurements) == 4

    measurement_configs = measurements[0]['config']

    assert 'parameters' in measurement_configs
    assert len(measurement_configs['parameters']) == 3
    parnames = [p['name'] for p in measurement_configs['parameters']]
    assert sorted(parnames) == sorted(['lumi', 'SigXsecOverSM', 'syst1'])

    lumi_param_config = measurement_configs['parameters'][0]
    assert 'auxdata' in lumi_param_config
    assert lumi_param_config['auxdata'] == [1.0]
    assert 'bounds' in lumi_param_config
    assert lumi_param_config['bounds'] == [[0.5, 1.5]]
    assert 'inits' in lumi_param_config
    assert lumi_param_config['inits'] == [1.0]
    assert 'sigmas' in lumi_param_config
    assert lumi_param_config['sigmas'] == [0.1]


@pytest.mark.parametrize("const", ['False', 'True'])
def test_spaces_in_measurement_config(const):
    toplvl = ET.Element("Combination")
    meas = ET.Element(
        "Measurement",
        Name='NormalMeasurement',
        Lumi=str(1.0),
        LumiRelErr=str(0.017),
        ExportOnly=str(True),
    )
    poiel = ET.Element('POI')
    poiel.text = 'mu_SIG '  # space
    meas.append(poiel)

    setting = ET.Element('ParamSetting', Const=const)
    setting.text = ' '.join(['Lumi', 'alpha_mu_both']) + ' '  # spacces
    meas.append(setting)

    toplvl.append(meas)

    meas_json = pyhf.readxml.process_measurements(toplvl)[0]
    assert meas_json['config']['poi'] == 'mu_SIG'
    assert [x['name'] for x in meas_json['config']['parameters']] == ['lumi', 'mu_both']


@pytest.mark.parametrize("const", ['False', 'True'])
def test_import_measurement_gamma_bins(const):
    toplvl = ET.Element("Combination")
    meas = ET.Element(
        "Measurement",
        Name='NormalMeasurement',
        Lumi=str(1.0),
        LumiRelErr=str(0.017),
        ExportOnly=str(True),
    )
    poiel = ET.Element('POI')
    poiel.text = 'mu_SIG'
    meas.append(poiel)

    setting = ET.Element('ParamSetting', Const=const)
    setting.text = ' '.join(['Lumi', 'alpha_mu_both', 'gamma_bin_0'])
    meas.append(setting)

    setting = ET.Element('ParamSetting', Val='2.0')
    setting.text = ' '.join(['gamma_bin_0'])
    meas.append(setting)

    toplvl.append(meas)

    with pytest.raises(ValueError):
        pyhf.readxml.process_measurements(toplvl)


@pytest.mark.parametrize(
    "configfile,rootdir",
    [
        (
            'validation/xmlimport_input/config/example.xml',
            'validation/xmlimport_input/',
        ),
        (
            'validation/xmlimport_input4/config/example.xml',
            'validation/xmlimport_input4/',
        ),
    ],
    ids=['xmlimport_input', 'xmlimport_input_histoPath'],
)
def test_import_prepHistFactory(configfile, rootdir):
    parsed_xml = pyhf.readxml.parse(configfile, rootdir)

    # build the spec, strictly checks properties included
    spec = {
        'channels': parsed_xml['channels'],
        'parameters': parsed_xml['measurements'][0]['config']['parameters'],
    }
    pdf = pyhf.Model(spec, poi_name='SigXsecOverSM')

    data = [
        binvalue
        for k in pdf.spec['channels']
        for binvalue in next(
            obs for obs in parsed_xml['observations'] if obs['name'] == k['name']
        )['data']
    ] + pdf.config.auxdata

    channels = {channel['name'] for channel in pdf.spec['channels']}
    samples = {
        channel['name']: [sample['name'] for sample in channel['samples']]
        for channel in pdf.spec['channels']
    }

    ###
    # signal overallsys
    # bkg1 overallsys (stat ignored)
    # bkg2 stateror (2 bins)
    # bkg2 overallsys

    assert 'channel1' in channels
    assert 'signal' in samples['channel1']
    assert 'background1' in samples['channel1']
    assert 'background2' in samples['channel1']

    assert pdf.spec['channels'][0]['samples'][1]['modifiers'][0]['type'] == 'lumi'
    assert pdf.spec['channels'][0]['samples'][2]['modifiers'][0]['type'] == 'lumi'

    assert pdf.spec['channels'][0]['samples'][2]['modifiers'][1]['type'] == 'staterror'
    assert pdf.spec['channels'][0]['samples'][2]['modifiers'][1]['data'] == [0, 10.0]

    assert pdf.spec['channels'][0]['samples'][1]['modifiers'][1]['type'] == 'staterror'
    assert all(
        np.isclose(
            pdf.spec['channels'][0]['samples'][1]['modifiers'][1]['data'], [5.0, 0.0]
        )
    )

    assert pdf.expected_actualdata(
        pyhf.tensorlib.astensor(pdf.config.suggested_init())
    ).tolist() == [120.0, 110.0]

    assert pdf.config.auxdata_order == [
        'lumi',
        'syst2',
        'syst3',
        'syst1',
        'staterror_channel1',
    ]

    assert data == [122.0, 112.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0]

    pars = pdf.config.suggested_init()
    pars[pdf.config.par_slice('SigXsecOverSM')] = [2.0]
    assert pdf.expected_data(pars, include_auxdata=False).tolist() == [140, 120]


def test_import_histosys():
    parsed_xml = pyhf.readxml.parse(
        'validation/xmlimport_input2/config/example.xml', 'validation/xmlimport_input2'
    )

    # build the spec, strictly checks properties included
    spec = {
        'channels': parsed_xml['channels'],
        'parameters': parsed_xml['measurements'][0]['config']['parameters'],
    }
    pdf = pyhf.Model(spec, poi_name='SigXsecOverSM')

    channels = {channel['name']: channel for channel in pdf.spec['channels']}

    assert channels['channel2']['samples'][0]['modifiers'][0]['type'] == 'lumi'
    assert channels['channel2']['samples'][0]['modifiers'][1]['type'] == 'histosys'


def test_import_filecache(mocker):
    mocker.patch("pyhf.readxml.uproot.open", wraps=uproot.open)

    pyhf.readxml.clear_filecache()

    parsed_xml = pyhf.readxml.parse(
        'validation/xmlimport_input/config/example.xml', 'validation/xmlimport_input/'
    )

    # call a second time (file should be cached now)
    parsed_xml2 = pyhf.readxml.parse(
        'validation/xmlimport_input/config/example.xml', 'validation/xmlimport_input/'
    )

    # check if uproot.open was only called once with the expected root file
    pyhf.readxml.uproot.open.assert_called_once_with(
        str(Path("validation/xmlimport_input").joinpath("./data/example.root"))
    )

    assert_equal_dictionary(parsed_xml, parsed_xml2)


def test_import_shapesys():
    parsed_xml = pyhf.readxml.parse(
        'validation/xmlimport_input3/config/examples/example_ShapeSys.xml',
        'validation/xmlimport_input3',
    )

    # build the spec, strictly checks properties included
    spec = {
        'channels': parsed_xml['channels'],
        'parameters': parsed_xml['measurements'][0]['config']['parameters'],
    }
    pdf = pyhf.Model(spec, poi_name='SigXsecOverSM')

    channels = {channel['name']: channel for channel in pdf.spec['channels']}

    assert channels['channel1']['samples'][1]['modifiers'][0]['type'] == 'lumi'
    assert channels['channel1']['samples'][1]['modifiers'][1]['type'] == 'shapesys'
    # NB: assert that relative uncertainty is converted to absolute uncertainty for shapesys
    assert channels['channel1']['samples'][1]['data'] == pytest.approx([100.0, 1.0e-4])
    assert channels['channel1']['samples'][1]['modifiers'][1]['data'] == pytest.approx(
        [10.0, 1.5e-5]
    )


def test_import_normfactor_bounds():
    parsed_xml = pyhf.readxml.parse(
        'validation/xmlimport_input2/config/example.xml', 'validation/xmlimport_input2'
    )

    ws = pyhf.Workspace(parsed_xml)
    assert ('SigXsecOverSM', 'normfactor') in ws.modifiers
    parameters = [
        p
        for p in ws.get_measurement(measurement_name='GaussExample')['config'][
            'parameters'
        ]
        if p['name'] == 'SigXsecOverSM'
    ]
    assert len(parameters) == 1
    parameter = parameters[0]
    assert parameter['bounds'] == [[0, 10]]


def test_import_shapefactor():
    parsed_xml = pyhf.readxml.parse(
        'validation/xmlimport_input/config/examples/example_DataDriven.xml',
        'validation/xmlimport_input',
    )

    # build the spec, strictly checks properties included
    spec = {
        'channels': parsed_xml['channels'],
        'parameters': parsed_xml['measurements'][0]['config']['parameters'],
    }
    pdf = pyhf.Model(spec, poi_name='SigXsecOverSM')

    channels = {channel['name']: channel for channel in pdf.spec['channels']}

    assert channels['controlRegion']['samples'][0]['modifiers'][0]['type'] == 'lumi'
    assert (
        channels['controlRegion']['samples'][0]['modifiers'][1]['type'] == 'staterror'
    )
    assert channels['controlRegion']['samples'][0]['modifiers'][2]['type'] == 'normsys'
    assert (
        channels['controlRegion']['samples'][1]['modifiers'][0]['type'] == 'shapefactor'
    )


def test_process_modifiers(mocker, caplog):
    sample = ET.Element(
        "Sample", Name='testSample', HistoPath="", HistoName="testSample"
    )
    normfactor = ET.Element(
        'NormFactor', Name="myNormFactor", Val='1', Low="0", High="3"
    )
    histosys = ET.Element(
        'HistoSys', Name='myHistoSys', HistoNameHigh='', HistoNameLow=''
    )
    normsys = ET.Element('OverallSys', Name='myNormSys', High='1.05', Low='0.95')
    shapesys = ET.Element('ShapeSys', Name='myShapeSys', HistoName='')
    shapefactor = ET.Element(
        "ShapeFactor",
        Name='myShapeFactor',
    )
    staterror = ET.Element('StatError', Activate='True')
    unknown_modifier = ET.Element('UnknownSys')

    sample.append(normfactor)
    sample.append(histosys)
    sample.append(normsys)
    sample.append(shapesys)
    sample.append(shapefactor)
    sample.append(staterror)
    sample.append(unknown_modifier)

    _data = [0.0]
    _err = [1.0]
    mocker.patch('pyhf.readxml.import_root_histogram', return_value=(_data, _err))
    with caplog.at_level(logging.DEBUG, 'pyhf.readxml'):
        result = pyhf.readxml.process_sample(sample, '', '', '', 'myChannel')

    assert "not considering modifier tag <Element 'UnknownSys'" in caplog.text
    assert len(result['modifiers']) == 6
    assert {'name': 'myNormFactor', 'type': 'normfactor', 'data': None} in result[
        'modifiers'
    ]
    assert {
        'name': 'myHistoSys',
        'type': 'histosys',
        'data': {'lo_data': _data, 'hi_data': _data},
    } in result['modifiers']
    assert {
        'name': 'myNormSys',
        'type': 'normsys',
        'data': {'lo': 0.95, 'hi': 1.05},
    } in result['modifiers']
    assert {'name': 'myShapeSys', 'type': 'shapesys', 'data': _data} in result[
        'modifiers'
    ]
    assert {'name': 'myShapeFactor', 'type': 'shapefactor', 'data': None} in result[
        'modifiers'
    ]
    assert {'name': 'staterror_myChannel', 'type': 'staterror', 'data': _err} in result[
        'modifiers'
    ]


def test_import_validation_exception(mocker, caplog):
    mocker.patch(
        'pyhf.schema.validate',
        side_effect=pyhf.exceptions.InvalidSpecification(
            ValidationError('this is an invalid specification')
        ),
    )

    with caplog.at_level(logging.WARNING, "pyhf.readxml"):
        pyhf.readxml.parse(
            'validation/xmlimport_input2/config/example.xml',
            'validation/xmlimport_input2',
            validation_as_error=False,
        )
        assert "this is an invalid specification" in caplog.text

    with pytest.raises(pyhf.exceptions.InvalidSpecification):
        pyhf.readxml.parse(
            'validation/xmlimport_input2/config/example.xml',
            'validation/xmlimport_input2',
            validation_as_error=True,
        )


def test_import_noChannelData(mocker, datadir):
    _data = [0.0]
    _err = [1.0]
    mocker.patch('pyhf.readxml.import_root_histogram', return_value=(_data, _err))

    basedir = datadir.joinpath("xmlimport_noChannelData")
    with pytest.raises(
        RuntimeError, match="Channel channel1 is missing data. See issue #1911"
    ):
        pyhf.readxml.parse(basedir.joinpath("config/example.xml"), basedir)


def test_import_noChannelDataPaths(mocker, datadir):
    _data = [0.0]
    _err = [1.0]
    mocker.patch('pyhf.readxml.import_root_histogram', return_value=(_data, _err))

    basedir = datadir.joinpath("xmlimport_noChannelDataPaths")
    with pytest.raises(NotImplementedError) as excinfo:
        pyhf.readxml.parse(basedir.joinpath("config/example.xml"), basedir)
    assert (
        "Conversion of workspaces without data is currently not supported.\nSee https://github.com/scikit-hep/pyhf/issues/566"
        in str(excinfo.value)
    )


def test_import_missingPOI(mocker, datadir):
    _data = [0.0]
    _err = [1.0]
    mocker.patch('pyhf.readxml.import_root_histogram', return_value=(_data, _err))

    basedir = datadir.joinpath("xmlimport_missingPOI")
    with pytest.raises(
        RuntimeError, match="Measurement GaussExample is missing POI specification"
    ):
        pyhf.readxml.parse(basedir.joinpath("config/example.xml"), basedir)


def test_import_resolver():
    rootdir = Path('/current/working/dir')
    mounts = [(Path('/this/path/changed'), Path('/my/abs/path'))]
    resolver = pyhf.readxml.resolver_factory(rootdir, mounts)

    assert resolver('relative/path') == Path('/current/working/dir/relative/path')
    assert resolver('relative/path/') == Path('/current/working/dir/relative/path')
    assert resolver('relative/path/to/file.txt') == Path(
        '/current/working/dir/relative/path/to/file.txt'
    )
    assert resolver('/absolute/path') == Path('/absolute/path')
    assert resolver('/absolute/path/') == Path('/absolute/path')
    assert resolver('/absolute/path/to/file.txt') == Path('/absolute/path/to/file.txt')
    assert resolver('/my/abs/path') == Path('/this/path/changed')
    assert resolver('/my/abs/path/') == Path('/this/path/changed')
    assert resolver('/my/abs/path/to/file.txt') == Path(
        '/this/path/changed/to/file.txt'
    )

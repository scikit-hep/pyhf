import pyhf
import pyhf.readxml
import numpy as np
import uproot
from pathlib import Path
import pytest
import xml.etree.cElementTree as ET


def assert_equal_dictionary(d1, d2):
    "recursively compare 2 dictionaries"
    for k in d1.keys():
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
    with pytest.raises(RuntimeError) as excinfo:
        pyhf.readxml.dedupe_parameters(parameters)
        assert 'SigXsecOverSM' in str(excinfo.value)


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
    setting.text = ' '.join(['Lumi', 'mu_both', 'mu_paramSettingOnly'])
    meas.append(setting)

    setting = ET.Element('ParamSetting', Val='2.0')
    setting.text = ' '.join(['mu_both'])
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
    setting.text = ' '.join(['mu_both'])
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
    assert sorted(parnames) == sorted(['lumi', 'SigXsecOverSM', 'alpha_syst1'])

    lumi_param_config = measurement_configs['parameters'][0]
    assert 'auxdata' in lumi_param_config
    assert lumi_param_config['auxdata'] == [1.0]
    assert 'bounds' in lumi_param_config
    assert lumi_param_config['bounds'] == [[0.5, 1.5]]
    assert 'inits' in lumi_param_config
    assert lumi_param_config['inits'] == [1.0]
    assert 'sigmas' in lumi_param_config
    assert lumi_param_config['sigmas'] == [0.1]


def test_import_prepHistFactory():
    parsed_xml = pyhf.readxml.parse(
        'validation/xmlimport_input/config/example.xml', 'validation/xmlimport_input/'
    )

    # build the spec, strictly checks properties included
    spec = {
        'channels': parsed_xml['channels'],
        'parameters': parsed_xml['measurements'][0]['config']['parameters'],
    }
    pdf = pyhf.Model(spec, poiname='SigXsecOverSM')

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

    assert pdf.config.auxdata_order == sorted(
        ['lumi', 'syst1', 'staterror_channel1', 'syst2', 'syst3']
    )

    assert data == [122.0, 112.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]

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
    pdf = pyhf.Model(spec, poiname='SigXsecOverSM')

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
        Path().joinpath("validation/xmlimport_input", "./data/example.root").as_posix()
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
    pdf = pyhf.Model(spec, poiname='SigXsecOverSM')

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

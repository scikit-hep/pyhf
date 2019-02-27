import pyhf
import pyhf.readxml
import numpy as np
import uproot
import os
import pytest


def assert_equal_dictionary(d1, d2):
    "recursively compare 2 dictionaries"
    for k in d1.keys():
        assert k in d2
        if type(d1[k]) is dict:
            assert_equal_dictionary(d1[k], d2[k])
        else:
            assert d1[k] == d2[k]


def test_import_measurements():
    parsed_xml = pyhf.readxml.parse(
        'validation/xmlimport_input/config/example.xml', 'validation/xmlimport_input/'
    )
    assert 'toplvl' in parsed_xml
    assert 'measurements' in parsed_xml['toplvl']

    measurements = parsed_xml['toplvl']['measurements']
    assert len(measurements) == 4

    measurement_configs = measurements[0]['config']

    assert 'parameters' in measurement_configs
    assert len(measurement_configs['parameters']) == 2
    assert measurement_configs['parameters'][0]['name'] == 'lumi'
    assert measurement_configs['parameters'][1]['name'] == 'alpha_syst1'

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
        'parameters': parsed_xml['toplvl']['measurements'][0]['config']['parameters'],
    }
    pdf = pyhf.Model(spec, poiname='SigXsecOverSM')

    data = [
        binvalue
        for k in pdf.spec['channels']
        for binvalue in parsed_xml['data'][k['name']]
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

    assert pdf.expected_actualdata(pdf.config.suggested_init()).tolist() == [
        120.0,
        110.0,
    ]

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
        'parameters': parsed_xml['toplvl']['measurements'][0]['config']['parameters'],
    }
    pdf = pyhf.Model(spec, poiname='SigXsecOverSM')

    data = [
        binvalue
        for k in pdf.spec['channels']
        for binvalue in parsed_xml['data'][k['name']]
    ] + pdf.config.auxdata

    channels = {channel['name']: channel for channel in pdf.spec['channels']}
    samples = {
        channel['name']: [sample['name'] for sample in channel['samples']]
        for channel in pdf.spec['channels']
    }

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
        os.path.join("validation/xmlimport_input", "./data/example.root")
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
        'parameters': parsed_xml['toplvl']['measurements'][0]['config']['parameters'],
    }
    pdf = pyhf.Model(spec, poiname='SigXsecOverSM')

    data = [
        binvalue
        for k in pdf.spec['channels']
        for binvalue in parsed_xml['data'][k['name']]
    ] + pdf.config.auxdata

    channels = {channel['name']: channel for channel in pdf.spec['channels']}
    samples = {
        channel['name']: [sample['name'] for sample in channel['samples']]
        for channel in pdf.spec['channels']
    }

    assert channels['channel1']['samples'][1]['modifiers'][0]['type'] == 'lumi'
    assert channels['channel1']['samples'][1]['modifiers'][1]['type'] == 'shapesys'
    # NB: assert that relative uncertainty is converted to absolute uncertainty for shapesys
    assert channels['channel1']['samples'][1]['data'] == pytest.approx([100.0, 1.0e-4])
    assert channels['channel1']['samples'][1]['modifiers'][1]['data'] == pytest.approx(
        [10.0, 1.5e-5]
    )

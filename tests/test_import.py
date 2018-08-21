import pyhf
import pyhf.readxml
import json
import jsonschema
import pytest
import pkg_resources 

@pytest.fixture(scope='module')
def schema():
    return json.load(open(pkg_resources.resource_filename('pyhf','data/spec.json')))

def test_import_prepHistFactory(schema):
    parsed_xml = pyhf.readxml.parse('validation/xmlimport_input/config/example.xml',
                              'validation/xmlimport_input/')

    # build the spec, strictly checks properties included
    spec = {'channels': parsed_xml['channels']}
    jsonschema.validate(spec, schema)
    pdf = pyhf.hfpdf(spec, poiname='SigXsecOverSM')

    data = [binvalue for k in pdf.spec['channels'] for binvalue
            in parsed_xml['data'][k['name']]] + pdf.config.auxdata

    channels = {channel['name'] for channel in pdf.spec['channels']}
    samples = {channel['name']: [sample['name'] for sample in channel['samples']] for channel in pdf.spec['channels']}

    assert data == [122.0, 112.0, 0, 0, 1.0, 1.0, 0.0]
    ###
    ### signal overallsys
    ### bkg1 overallsys (stat ignored)
    ### bkg2 stateror (2 bins)
    ### bkg2 overallsys

    assert 'channel1' in channels
    assert 'signal' in samples['channel1']
    assert 'background1' in samples['channel1']
    assert 'background2' in samples['channel1']

    assert pdf.expected_actualdata(
        pdf.config.suggested_init()).tolist() == [120.0, 110.0]

    pars = pdf.config.suggested_init()
    pars[pdf.config.par_slice('SigXsecOverSM')] = [2.0]
    assert pdf.expected_data(
        pars, include_auxdata=False).tolist() == [140, 120]

def test_import_histosys(schema):
    parsed_xml = pyhf.readxml.parse('validation/xmlimport_input2/config/example.xml',
                              'validation/xmlimport_input2')

    # build the spec, strictly checks properties included
    spec = {'channels': parsed_xml['channels']}
    jsonschema.validate(spec, schema)
    pdf = pyhf.hfpdf(spec, poiname='SigXsecOverSM')

    data = [binvalue for k in pdf.spec['channels'] for binvalue
            in parsed_xml['data'][k['name']]] + pdf.config.auxdata

    channels = {channel['name']:channel for channel in pdf.spec['channels']}
    samples = {channel['name']: [sample['name'] for sample in channel['samples']] for channel in pdf.spec['channels']}

    assert channels['channel2']['samples'][0]['modifiers'][0]['type'] == 'histosys'

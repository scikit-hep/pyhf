import pyhf
import pyhf.readxml
import json
import pytest
import numpy as np

def test_import_prepHistFactory():
    parsed_xml = pyhf.readxml.parse('validation/xmlimport_input/config/example.xml',
                              'validation/xmlimport_input/')

    # build the spec, strictly checks properties included
    spec = {'channels': parsed_xml['channels']}
    pdf = pyhf.Model(spec, poiname='SigXsecOverSM')

    data = [binvalue for k in pdf.spec['channels'] for binvalue
            in parsed_xml['data'][k['name']]] + pdf.config.auxdata

    channels = {channel['name'] for channel in pdf.spec['channels']}
    samples = {channel['name']: [sample['name'] for sample in channel['samples']] for channel in pdf.spec['channels']}


    ###
    ### signal overallsys
    ### bkg1 overallsys (stat ignored)
    ### bkg2 stateror (2 bins)
    ### bkg2 overallsys

    assert 'channel1' in channels
    assert 'signal' in samples['channel1']
    assert 'background1' in samples['channel1']
    assert 'background2' in samples['channel1']

    assert pdf.spec['channels'][0]['samples'][2]['modifiers'][0]['type'] == 'staterror'
    assert pdf.spec['channels'][0]['samples'][2]['modifiers'][0]['data'] == [0,10.]

    assert pdf.spec['channels'][0]['samples'][1]['modifiers'][0]['type'] == 'staterror'
    assert all(np.isclose(pdf.spec['channels'][0]['samples'][1]['modifiers'][0]['data'],[5.0, 0.0]))

    assert pdf.expected_actualdata(
        pdf.config.suggested_init()).tolist() == [120.0, 110.0]

    assert pdf.config.auxdata_order ==  ['syst1', 'staterror_channel1', 'syst2', 'syst3']

    assert data == [122.0, 112.0, 0.0, 1.0, 1.0, 0.0, 0.0]

    pars = pdf.config.suggested_init()
    pars[pdf.config.par_slice('SigXsecOverSM')] = [2.0]
    assert pdf.expected_data(
        pars, include_auxdata=False).tolist() == [140, 120]

def test_import_histosys():
    parsed_xml = pyhf.readxml.parse('validation/xmlimport_input2/config/example.xml',
                              'validation/xmlimport_input2')

    # build the spec, strictly checks properties included
    spec = {'channels': parsed_xml['channels']}
    pdf = pyhf.Model(spec, poiname='SigXsecOverSM')

    data = [binvalue for k in pdf.spec['channels'] for binvalue
            in parsed_xml['data'][k['name']]] + pdf.config.auxdata

    channels = {channel['name']:channel for channel in pdf.spec['channels']}
    samples = {channel['name']: [sample['name'] for sample in channel['samples']] for channel in pdf.spec['channels']}

    assert channels['channel2']['samples'][0]['modifiers'][0]['type'] == 'histosys'

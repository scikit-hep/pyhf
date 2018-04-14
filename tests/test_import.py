import pyhf
import pyhf.readxml
import json
import jsonschema

def test_import_prepHistFactory():
    schema = json.load(open('validation/spec.json'))
    spec = pyhf.readxml.parse('validation/xmlimport_input/config/example.xml',
                              'validation/xmlimport_input/')

    jsonschema.validate(spec['channels'], schema)
    pdf = pyhf.hfpdf(spec['channels'], poiname='SigXsecOverSM')

    data = [binvalue for k in pdf.config.channel_order for binvalue
            in spec['data'][k]] + pdf.config.auxdata

    channels = {channel['name'] for channel in spec['channels']}
    samples = {'channel1': [sample['name'] for sample in spec['channels']['channel1']['samples']]}

    assert data == [122.0, 112.0, 0, 0, 0]

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

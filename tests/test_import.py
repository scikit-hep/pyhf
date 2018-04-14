import pyhf
import pyhf.readxml


def test_import_prepHistFactory():
    spec = pyhf.readxml.parse('validation/xmlimport_input/config/example.xml',
                              'validation/xmlimport_input/')
    pdf = pyhf.hfpdf(spec['channels'], poiname='SigXsecOverSM')

    data = [binvalue for k in pdf.config.channel_order for binvalue
            in spec['data'][k]] + pdf.config.auxdata

    samples = {'channel1': [sample['name'] for sample in spec['channels']['channel1']['samples']]}

    assert data == [122.0, 112.0, 0, 0, 0]

    assert 'channel1' in spec['channels']
    assert 'signal' in samples['channel1']
    assert 'background1' in samples['channel1']
    assert 'background2' in samples['channel1']

    assert pdf.expected_actualdata(
        pdf.config.suggested_init()).tolist() == [120.0, 110.0]

    pars = pdf.config.suggested_init()
    pars[pdf.config.par_slice('SigXsecOverSM')] = [2.0]
    assert pdf.expected_data(
        pars, include_auxdata=False).tolist() == [140, 120]

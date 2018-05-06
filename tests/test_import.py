import pyhf
import pyhf.readxml
import json
import pytest

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

#@pytest.mark.slow
def test_import_multibin_multibjets():
    parsed_xml = pyhf.readxml.parse('validation/multibin_multibjets/config/NormalMeasurement.xml', 'validation/multibin_multibjets/')

    # build the spec, strictly checks properties included
    spec = {'channels': parsed_xml['channels']}
    pdf = pyhf.hfpdf(spec, poiname='mu_SIG')

    data = [binvalue for k in pdf.spec['channels'] for binvalue
            in parsed_xml['data'][k['name']]] + pdf.config.auxdata

    channels = {channel['name'] for channel in pdf.spec['channels']}
    samples = {channel['name']: [sample['name'] for sample in channel['samples']] for channel in pdf.spec['channels']}

    assert data == pytest.approx([2.242565631866455, 19.573698043823242, 35.0, 31.0, 2.6398260593414307, 24.0, 3.1293132305145264, 62.0, 14.0, 17.964872360229492, 60.0, 34.0, 13.589479446411133, 6.73526668548584, 56.0, 15.0, 42.0, 37.0, 22.0, 46.0, 62.0, 134.0, 44.0, 82.0, 248.0, 0.6610341668128967, 179.0, 26.24163246154785, 20.481304168701172, 0.9515645503997803, 36.0, 11.0, 107.13920593261719, 2.658748149871826, 1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 1.0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0, 1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0, 0, 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0, 0, 0, 1.0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0, 0, 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 1.0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0, 0, 0])

    assert 'SR_1l_Hnj_Lmeff_cuts' in channels
    assert len(channels) == 34
    assert samples['SR_1l_Hnj_Lmeff_cuts'] == ['ttbar', 'topEW', 'singletop', 'Gtt_2400_5000_800']
    assert [mod['name'] for mod in spec['channels'][0]['samples'][0]['modifiers']] == ['staterror_SR_1l_Hnj_Lmeff_cuts', 'ttbar_syst_SR_1l_Hnj_Lmeff', 'mu_ttbar_Hnj_Lmeff']

    assert pdf.expected_actualdata(
        pdf.config.suggested_init()).tolist() == pytest.approx([2.273263268172741, 19.594275111332536, 53.039544586092234, 39.44240501523018, 2.759223287925124, 26.539743864908814, 3.216025687754154, 84.0797322075814, 11.943474016617984, 17.987265093252063, 60.41570029221475, 41.746501334011555, 13.608307596296072, 6.927114307880402, 54.89945707144216, 14.585989810526371, 55.608559891581535, 51.10562594886869, 23.844748482108116, 57.99289866536856, 57.15544019266963, 133.34641349315643, 36.610900823026896, 77.52383244037628, 249.17225728160702, 1.625256523489952, 160.91590662300587, 26.2435217150487, 20.61478441953659, 1.097829520702362, 34.58026654832065, 16.083601087331772, 107.15127668529749, 3.825026348233223])

    pars = pdf.config.suggested_init()
    pars[pdf.config.par_slice('mu_SIG')] = [2.0]
    assert pdf.expected_data(
        pars, include_auxdata=False).tolist() == pytest.approx([2.303960859775543, 19.61485241726041, 53.056741178035736, 39.44240501523018, 2.8786206301301718, 26.559366576373577, 3.302738279104233, 84.08167374506593, 11.949640973471105, 18.009657438844442, 60.429331447929144, 41.770013108849525, 13.627135626971722, 7.1189620196819305, 54.90506408456713, 14.667295202612877, 55.614231429994106, 51.11959879659116, 23.904665499925613, 58.20833582431078, 57.169096760451794, 133.34641349315643, 36.65608770400286, 77.52383244037628, 249.17508681071922, 2.5894788950681686, 160.91590662300587, 26.24541132617742, 20.74826619029045, 1.2440944761037827, 34.58848659414798, 16.30122047662735, 107.1633462458849, 4.991304591298103])

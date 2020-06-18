import pyhf
import pyhf.writexml
import pytest
import json
import xml.etree.cElementTree as ET


def spec_staterror():
    spec = {
        'channels': [
            {
                'name': 'firstchannel',
                'samples': [
                    {
                        'name': 'mu',
                        'data': [10.0, 10.0],
                        'modifiers': [
                            {'name': 'mu', 'type': 'normfactor', 'data': None}
                        ],
                    },
                    {
                        'name': 'bkg1',
                        'data': [50.0, 70.0],
                        'modifiers': [
                            {
                                'name': 'stat_firstchannel',
                                'type': 'staterror',
                                'data': [12.0, 12.0],
                            }
                        ],
                    },
                    {
                        'name': 'bkg2',
                        'data': [30.0, 20.0],
                        'modifiers': [
                            {
                                'name': 'stat_firstchannel',
                                'type': 'staterror',
                                'data': [5.0, 5.0],
                            }
                        ],
                    },
                    {'name': 'bkg3', 'data': [20.0, 15.0], 'modifiers': []},
                ],
            }
        ]
    }
    return spec


def spec_histosys():
    source = json.load(open('validation/data/2bin_histosys_example2.json'))
    spec = {
        'channels': [
            {
                'name': 'singlechannel',
                'samples': [
                    {
                        'name': 'signal',
                        'data': source['bindata']['sig'],
                        'modifiers': [
                            {'name': 'mu', 'type': 'normfactor', 'data': None}
                        ],
                    },
                    {
                        'name': 'background',
                        'data': source['bindata']['bkg'],
                        'modifiers': [
                            {
                                'name': 'bkg_norm',
                                'type': 'histosys',
                                'data': {
                                    'lo_data': source['bindata']['bkgsys_dn'],
                                    'hi_data': source['bindata']['bkgsys_up'],
                                },
                            }
                        ],
                    },
                ],
            }
        ]
    }
    return spec


def spec_normsys():
    source = json.load(open('validation/data/2bin_histosys_example2.json'))
    spec = {
        'channels': [
            {
                'name': 'singlechannel',
                'samples': [
                    {
                        'name': 'signal',
                        'data': source['bindata']['sig'],
                        'modifiers': [
                            {'name': 'mu', 'type': 'normfactor', 'data': None}
                        ],
                    },
                    {
                        'name': 'background',
                        'data': source['bindata']['bkg'],
                        'modifiers': [
                            {
                                'name': 'bkg_norm',
                                'type': 'normsys',
                                'data': {'lo': 0.9, 'hi': 1.1},
                            }
                        ],
                    },
                ],
            }
        ]
    }
    return spec


def spec_shapesys():
    source = json.load(open('validation/data/2bin_histosys_example2.json'))
    spec = {
        'channels': [
            {
                'name': 'singlechannel',
                'samples': [
                    {
                        'name': 'signal',
                        'data': source['bindata']['sig'],
                        'modifiers': [
                            {'name': 'mu', 'type': 'normfactor', 'data': None}
                        ],
                    },
                    {
                        'name': 'background',
                        'data': source['bindata']['bkg'],
                        'modifiers': [
                            {'name': 'bkg_norm', 'type': 'shapesys', 'data': [10, 10]}
                        ],
                    },
                ],
            }
        ]
    }
    return spec


def test_export_measurement():
    measurementspec = {
        "config": {
            "parameters": [
                {
                    "auxdata": [1.0],
                    "bounds": [[0.855, 1.145]],
                    "inits": [1.0],
                    "name": "lumi",
                    "sigmas": [0.029],
                }
            ],
            "poi": "mu",
        },
        "name": "NormalMeasurement",
    }
    m = pyhf.writexml.build_measurement(measurementspec)
    assert m is not None
    assert m.attrib['Name'] == measurementspec['name']
    assert m.attrib['Lumi'] == str(
        measurementspec['config']['parameters'][0]['auxdata'][0]
    )
    assert m.attrib['LumiRelErr'] == str(
        measurementspec['config']['parameters'][0]['sigmas'][0]
    )
    poi = m.find('POI')
    assert poi is not None
    assert poi.text == measurementspec['config']['poi']
    paramsetting = m.find('ParamSetting')
    assert paramsetting is None


@pytest.mark.parametrize(
    "spec, has_root_data, attrs",
    [
        (spec_staterror(), True, ['Activate', 'HistoName']),
        (spec_histosys(), True, ['HistoNameHigh', 'HistoNameLow']),
        (spec_normsys(), False, ['High', 'Low']),
        (spec_shapesys(), True, ['ConstraintType', 'HistoName']),
    ],
    ids=['staterror', 'histosys', 'normsys', 'shapesys'],
)
def test_export_modifier(mocker, spec, has_root_data, attrs):
    channelspec = spec['channels'][0]
    channelname = channelspec['name']
    samplespec = channelspec['samples'][1]
    samplename = samplespec['name']
    sampledata = samplespec['data']
    modifierspec = samplespec['modifiers'][0]

    mocker.patch('pyhf.writexml._ROOT_DATA_FILE')
    modifier = pyhf.writexml.build_modifier(
        {'measurements': [{'config': {'parameters': []}}]},
        modifierspec,
        channelname,
        samplename,
        sampledata,
    )
    # if the modifier is a staterror, it has no Name
    if 'Name' in modifier.attrib:
        assert modifier.attrib['Name'] == modifierspec['name']
    assert all(attr in modifier.attrib for attr in attrs)
    assert pyhf.writexml._ROOT_DATA_FILE.__setitem__.called == has_root_data


@pytest.mark.parametrize(
    "spec, normfactor_config",
    [
        (spec_staterror(), dict(name='mu', inits=[1.0], bounds=[[0.0, 8.0]])),
        (spec_histosys(), dict()),
        (spec_normsys(), dict(name='mu', inits=[2.0], bounds=[[0.0, 10.0]])),
        (spec_shapesys(), dict(name='mu', inits=[1.0], bounds=[[5.0, 10.0]])),
    ],
    ids=['upper-bound', 'empty-config', 'init', 'lower-bound'],
)
def test_export_modifier_normfactor(mocker, spec, normfactor_config):
    channelspec = spec['channels'][0]
    channelname = channelspec['name']
    samplespec = channelspec['samples'][0]
    samplename = samplespec['name']
    sampledata = samplespec['data']
    modifierspec = samplespec['modifiers'][0]

    mocker.patch('pyhf.writexml._ROOT_DATA_FILE')
    modifier = pyhf.writexml.build_modifier(
        {
            'measurements': [
                {
                    'config': {
                        'parameters': [normfactor_config] if normfactor_config else []
                    }
                }
            ]
        },
        modifierspec,
        channelname,
        samplename,
        sampledata,
    )

    assert all(attr in modifier.attrib for attr in ['Name', 'Val', 'High', 'Low'])
    assert float(modifier.attrib['Val']) == normfactor_config.get('inits', [1.0])[0]
    assert (
        float(modifier.attrib['Low'])
        == normfactor_config.get('bounds', [[0.0, 10.0]])[0][0]
    )
    assert (
        float(modifier.attrib['High'])
        == normfactor_config.get('bounds', [[0.0, 10.0]])[0][1]
    )


@pytest.mark.parametrize(
    "spec",
    [spec_staterror(), spec_histosys(), spec_normsys(), spec_shapesys()],
    ids=['staterror', 'histosys', 'normsys', 'shapesys'],
)
def test_export_sample(mocker, spec):
    channelspec = spec['channels'][0]
    channelname = channelspec['name']
    samplespec = channelspec['samples'][1]

    mocker.patch('pyhf.writexml.build_modifier', return_value=ET.Element("Modifier"))
    mocker.patch('pyhf.writexml._ROOT_DATA_FILE')
    sample = pyhf.writexml.build_sample({}, samplespec, channelname)
    assert sample.attrib['Name'] == samplespec['name']
    assert sample.attrib['HistoName']
    assert sample.attrib['InputFile']
    assert sample.attrib['NormalizeByTheory'] == str(False)
    assert pyhf.writexml.build_modifier.called
    assert pyhf.writexml._ROOT_DATA_FILE.__setitem__.called


@pytest.mark.parametrize(
    "spec", [spec_staterror(), spec_shapesys()], ids=['staterror', 'shapesys']
)
def test_export_sample_zerodata(mocker, spec):
    channelspec = spec['channels'][0]
    channelname = channelspec['name']
    samplespec = channelspec['samples'][1]
    samplename = samplespec['name']
    sampledata = [0.0] * len(samplespec['data'])

    mocker.patch('pyhf.writexml._ROOT_DATA_FILE')
    # make sure no RuntimeWarning, https://stackoverflow.com/a/45671804
    with pytest.warns(None) as record:
        for modifierspec in samplespec['modifiers']:
            pyhf.writexml.build_modifier(
                {'measurements': [{'config': {'parameters': []}}]},
                modifierspec,
                channelname,
                samplename,
                sampledata,
            )
    assert not record.list


@pytest.mark.parametrize(
    "spec",
    [spec_staterror(), spec_histosys(), spec_normsys(), spec_shapesys()],
    ids=['staterror', 'histosys', 'normsys', 'shapesys'],
)
def test_export_channel(mocker, spec):
    channelspec = spec['channels'][0]

    mocker.patch('pyhf.writexml.build_data', return_value=ET.Element("Data"))
    mocker.patch('pyhf.writexml.build_sample', return_value=ET.Element("Sample"))
    mocker.patch('pyhf.writexml._ROOT_DATA_FILE')
    channel = pyhf.writexml.build_channel({}, channelspec, {})
    assert channel.attrib['Name'] == channelspec['name']
    assert channel.attrib['InputFile']
    assert pyhf.writexml.build_data.called is False
    assert pyhf.writexml.build_sample.called
    assert pyhf.writexml._ROOT_DATA_FILE.__setitem__.called is False


def test_export_data(mocker):
    channelname = 'channel'
    dataspec = [{'name': channelname, 'data': [0, 1, 2, 3]}]

    mocker.patch('pyhf.writexml._ROOT_DATA_FILE')
    data = pyhf.writexml.build_data(dataspec, channelname)
    assert data.attrib['HistoName']
    assert data.attrib['InputFile']
    assert pyhf.writexml._ROOT_DATA_FILE.__setitem__.called

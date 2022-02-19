import json
import logging
import xml.etree.ElementTree as ET

import pytest
import uproot

import pyhf
import pyhf.writexml


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
    with open(
        "validation/data/2bin_histosys_example2.json", encoding="utf-8"
    ) as source_file:
        source = json.load(source_file)
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
    with open(
        "validation/data/2bin_histosys_example2.json", encoding="utf-8"
    ) as source_file:
        source = json.load(source_file)
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
    with open(
        "validation/data/2bin_histosys_example2.json", encoding="utf-8"
    ) as source_file:
        source = json.load(source_file)
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


def spec_shapefactor():
    with open(
        "validation/data/2bin_histosys_example2.json", encoding="utf-8"
    ) as source_file:
        source = json.load(source_file)
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
                            {'name': 'bkg_norm', 'type': 'shapefactor', 'data': None}
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
                },
                {"name": "syst1", "fixed": True},
                {"name": "syst2", "fixed": True},
                {"name": "syst3", "fixed": True},
                {"name": "syst4", "fixed": True},
            ],
            "poi": "mu",
        },
        "name": "NormalMeasurement",
    }
    modifiertypes = {
        'mu': 'normfactor',
        'lumi': 'lumi',
        'syst1': 'normsys',
        'syst2': 'histosys',
        'syst3': 'shapesys',
        'syst4': 'staterror',
    }
    m = pyhf.writexml.build_measurement(measurementspec, modifiertypes)
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
    assert paramsetting is not None
    assert 'alpha_syst1' in paramsetting.text
    assert 'alpha_syst2' in paramsetting.text
    assert 'gamma_syst3' in paramsetting.text
    assert 'gamma_syst4' in paramsetting.text


@pytest.mark.parametrize(
    "spec, has_root_data, attrs, modtype",
    [
        (spec_staterror(), True, ['Activate', 'HistoName'], 'staterror'),
        (spec_histosys(), True, ['HistoNameHigh', 'HistoNameLow'], 'histosys'),
        (spec_normsys(), False, ['High', 'Low'], 'normsys'),
        (spec_shapesys(), True, ['ConstraintType', 'HistoName'], 'shapesys'),
        (spec_shapefactor(), False, [], 'shapefactor'),
    ],
    ids=['staterror', 'histosys', 'normsys', 'shapesys', 'shapefactor'],
)
def test_export_modifier(mocker, caplog, spec, has_root_data, attrs, modtype):
    channelspec = spec['channels'][0]
    channelname = channelspec['name']
    samplespec = channelspec['samples'][1]
    samplename = samplespec['name']
    sampledata = samplespec['data']
    modifierspec = samplespec['modifiers'][0]

    assert modifierspec['type'] == modtype

    mocker.patch('pyhf.writexml._ROOT_DATA_FILE')

    with caplog.at_level(logging.DEBUG, 'pyhf.writexml'):
        modifier = pyhf.writexml.build_modifier(
            {'measurements': [{'config': {'parameters': []}}]},
            modifierspec,
            channelname,
            samplename,
            sampledata,
        )
    assert "Skipping modifier" not in caplog.text

    # if the modifier is a staterror, it has no Name
    if modtype == 'staterror':
        assert 'Name' not in modifier.attrib
    else:
        assert modifier.attrib['Name'] == modifierspec['name']
    assert all(attr in modifier.attrib for attr in attrs)
    assert pyhf.writexml._ROOT_DATA_FILE.__setitem__.called == has_root_data


def test_export_bad_modifier(caplog):
    with caplog.at_level(logging.DEBUG, 'pyhf.writexml'):
        pyhf.writexml.build_modifier(
            {'measurements': [{'config': {'parameters': []}}]},
            {'name': 'fakeModifier', 'type': 'unknown-modifier'},
            'fakeChannel',
            'fakeSample',
            None,
        )
    assert "Skipping modifier fakeModifier(unknown-modifier)" in caplog.text


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
    for modifierspec in samplespec['modifiers']:
        pyhf.writexml.build_modifier(
            {'measurements': [{'config': {'parameters': []}}]},
            modifierspec,
            channelname,
            samplename,
            sampledata,
        )


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


def test_export_root_histogram(mocker, tmp_path):
    """
    Test that pyhf.writexml._export_root_histogram writes out a histogram
    in the manner that uproot is expecting and verifies this by reading
    the serialized file
    """
    mocker.patch("pyhf.writexml._ROOT_DATA_FILE", {})
    pyhf.writexml._export_root_histogram("hist", [0, 1, 2, 3, 4, 5, 6, 7, 8])

    with uproot.recreate(tmp_path.joinpath("test_export_root_histogram.root")) as file:
        file["hist"] = pyhf.writexml._ROOT_DATA_FILE["hist"]

    with uproot.open(
        tmp_path.joinpath("test_export_root_histogram.root"), encoding="utf-8"
    ) as file:
        assert file["hist"].values().tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8]
        assert file["hist"].axis().edges().tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert file["hist"].name == "hist"


def test_export_duplicate_hist_name(mocker):
    mocker.patch('pyhf.writexml._ROOT_DATA_FILE', new={'duplicate_name': True})

    with pytest.raises(KeyError):
        pyhf.writexml._export_root_histogram('duplicate_name', [0, 1, 2])


def test_integer_data(datadir, mocker):
    """
    Test that a spec with only integer data will be written correctly
    """
    with open(
        datadir.joinpath("workspace_integer_data.json"), encoding="utf-8"
    ) as spec_file:
        spec = json.load(spec_file)
    channel_spec = spec["channels"][0]
    mocker.patch("pyhf.writexml._ROOT_DATA_FILE")

    channel = pyhf.writexml.build_channel(spec, channel_spec, {})
    assert channel


@pytest.mark.parametrize(
    "fname,val,low,high",
    [
        ('workspace_no_parameter_inits.json', '1', '-5', '5'),
        ('workspace_no_parameter_bounds.json', '5', '0', '10'),
    ],
    ids=['no_inits', 'no_bounds'],
)
def test_issue1814(datadir, mocker, fname, val, low, high):
    with open(datadir / fname, encoding="utf-8") as spec_file:
        spec = json.load(spec_file)

    modifierspec = {'data': None, 'name': 'mu_sig', 'type': 'normfactor'}
    channelname = None
    samplename = None
    sampledata = None

    modifier = pyhf.writexml.build_modifier(
        spec, modifierspec, channelname, samplename, sampledata
    )
    assert modifier is not None
    assert sorted(modifier.keys()) == ['High', 'Low', 'Name', 'Val']
    assert modifier.get('Val') == val
    assert modifier.get('Low') == low
    assert modifier.get('High') == high

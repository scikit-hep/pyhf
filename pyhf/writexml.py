import logging

import os
import xml.etree.cElementTree as ET
import numpy as np
import uproot
from uproot_methods.classes import TH1

_ROOT_DATA_FILE = None
_HISTNAME = "h{sample}{modifier}{highlow}_{channel}_obs_cuts"

log = logging.getLogger(__name__)


def export_root_histogram(histname, data):
    h = TH1.from_numpy((np.asarray(data), np.arange(len(data) + 1)))
    h._fName = histname
    # NB: uproot crashes for some reason, figure out why later
    # if histname in _ROOT_DATA_FILE:
    #    raise KeyError('Duplicate key {0} being written.'.format(histname))
    _ROOT_DATA_FILE[histname] = h


# https://stackoverflow.com/a/4590052
def indent(elem, level=0):
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def build_measurement(measurementspec):
    config = measurementspec['config']
    name = measurementspec['name']
    poi = config['poi']

    # we want to know which parameters are fixed (constant)
    # and to additionally extract the luminosity information
    fixed_params = []
    lumi = 1.0
    lumierr = 0.0
    for parameter in config['parameters']:
        if parameter['fixed']:
            pname = parameter['name']
            if pname == 'lumi':
                fixed_params.append('Lumi')
            else:
                fixed_params.append(pname)
        # we found luminosity, so handle it
        if parameter['name'] == 'lumi':
            lumi = parameter['auxdata'][0]
            lumierr = parameter['sigmas'][0]

    # define measurement
    meas = ET.Element("Measurement", Name=name, Lumi=str(lumi), LumiRelErr=str(lumierr))
    poiel = ET.Element('POI')
    poiel.text = poi
    meas.append(poiel)

    # add fixed parameters (constant)
    se = ET.Element('ParamSetting', Const='True')
    se.text = ' '.join(fixed_params)
    meas.append(se)
    return meas


def build_modifier(modifierspec, channelname, samplename):
    if modifierspec['name'] == 'lumi':
        return None
    mod_map = {
        'histosys': 'HistoSys',
        'staterror': 'StatError',
        'normsys': 'OverallSys',
        'shapesys': 'ShapeSys',
        'normfactor': 'NormFactor',
        'shapefactor': 'ShapeFactor',
    }

    fmtvars = {
        'channel': channelname,
        'sample': samplename,
        'modifier': modifierspec['name'],
    }
    attrs = {'Name': modifierspec['name']}
    if modifierspec['type'] == 'histosys':
        attrs['HistoNameLow'] = _HISTNAME.format(highlow='Low', **fmtvars)
        attrs['HistoNameHigh'] = _HISTNAME.format(highlow='High', **fmtvars)
        export_root_histogram(attrs['HistoNameLow'], modifierspec['data']['lo_data'])
        export_root_histogram(attrs['HistoNameHigh'], modifierspec['data']['hi_data'])
    elif modifierspec['type'] == 'normsys':
        attrs['High'] = str(modifierspec['data']['hi'])
        attrs['Low'] = str(modifierspec['data']['lo'])
    elif modifierspec['type'] == 'normfactor':
        attrs['Val'] = '1'
        attrs['High'] = '10'
        attrs['Low'] = '0'
    elif modifierspec['type'] == 'staterror':
        attrs['Activate'] = 'True'
    else:
        log.warning(
            'Skipping {0}({1}) for now'.format(
                modifierspec['name'], modifierspec['type']
            )
        )

    modifier = ET.Element(mod_map[modifierspec['type']], **attrs)
    return modifier


def build_sample(samplespec, channelname):
    fmtvars = {
        'channel': channelname,
        'sample': samplespec['name'],
        'modifier': '',
        'highlow': '',
    }
    histname = _HISTNAME.format(**fmtvars)
    sample = ET.Element(
        'Sample',
        Name=samplespec['name'],
        HistoName=histname,
        InputFile=_ROOT_DATA_FILE._path,
    )
    for modspec in samplespec['modifiers']:
        modifier = build_modifier(modspec, channelname, samplespec['name'])
        if modifier is not None:
            sample.append(modifier)
    export_root_histogram(histname, samplespec['data'])
    return sample


def build_channel(channelspec):
    channel = ET.Element(
        'Channel', Name=channelspec['name'], InputFile=_ROOT_DATA_FILE._path
    )
    for samplespec in channelspec['samples']:
        channel.append(build_sample(samplespec, channelspec['name']))
    return channel


def writexml(spec, specdir, data_rootdir, result_outputprefix):
    global _ROOT_DATA_FILE

    combination = ET.Element("Combination", OutputFilePrefix=result_outputprefix)

    with uproot.recreate(os.path.join(data_rootdir, 'data.root')) as _ROOT_DATA_FILE:
        for channelspec in spec['channels']:
            channelfilename = os.path.join(
                specdir, 'channel_{}.xml'.format(channelspec['name'])
            )
            with open(channelfilename, 'w') as channelfile:
                channel = build_channel(channelspec)
                indent(channel)
                channelfile.write(
                    ET.tostring(channel, encoding='utf-8').decode('utf-8')
                )

            inp = ET.Element("Input")
            inp.text = channelfilename
            combination.append(inp)

    for measurement in spec['toplvl']['measurements']:
        combination.append(build_measurement(measurement))
    indent(combination)
    return ET.tostring(combination, encoding='utf-8')

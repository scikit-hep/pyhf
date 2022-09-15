import logging

from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import numpy as np

import uproot

from pyhf.mixins import _ChannelSummaryMixin
from pyhf.schema import path as schema_path

_ROOT_DATA_FILE = None

log = logging.getLogger(__name__)

__all__ = [
    "build_channel",
    "build_data",
    "build_measurement",
    "build_modifier",
    "build_sample",
    "indent",
]


def __dir__():
    return __all__


# 'spec' gets passed through all functions as NormFactor is a unique case of having
# parameter configurations stored at the modifier-definition-spec level. This means
# that build_modifier() needs access to the measurements. The call stack is:
#
#      writexml
#          ->build_channel
#              ->build_sample
#                  ->build_modifier
#
#  Therefore, 'spec' needs to be threaded through all these calls.


def _make_hist_name(channel, sample, modifier='', prefix='hist', suffix=''):
    middle = '_'.join(filter(lambda x: x, [channel, sample, modifier]))
    return f"{prefix}{middle}{suffix}"


def _export_root_histogram(hist_name, data):
    if hist_name in _ROOT_DATA_FILE:
        raise KeyError(f"Duplicate key {hist_name} being written.")
    _ROOT_DATA_FILE[hist_name] = uproot.to_writable(
        (np.asarray(data), np.arange(len(data) + 1))
    )


# https://stackoverflow.com/a/4590052
def indent(elem, level=0):
    i = "\n" + level * "  "
    if elem:
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for subelem in elem:
            indent(subelem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def build_measurement(measurementspec, modifiertypes):
    """
    Build the XML measurement specification for a given measurement adhering to defs.json/#definitions/measurement.

    Args:
        measurementspec (:obj:`dict`): The measurements specification from a :class:`~pyhf.workspace.Workspace`.
        modifiertypes (:obj:`dict`): A mapping from modifier name (:obj:`str`) to modifier type (:obj:`str`).

    Returns:
        :class:`xml.etree.cElementTree.Element`: The XML measurement specification.

    """
    # need to determine prefixes
    prefixes = {
        'normsys': 'alpha_',
        'histosys': 'alpha_',
        'shapesys': 'gamma_',
        'staterror': 'gamma_',
    }

    config = measurementspec['config']
    name = measurementspec['name']
    poi = config['poi']

    # we want to know which parameters are fixed (constant)
    # and to additionally extract the luminosity information
    fixed_params = []
    lumi = 1.0
    lumierr = 0.0
    for parameter in config['parameters']:
        if parameter.get('fixed', False):
            pname = parameter['name']
            if pname == 'lumi':
                fixed_params.append('Lumi')
            else:
                prefix = prefixes.get(modifiertypes[pname], '')
                fixed_params.append(f'{prefix}{pname}')
        # we found luminosity, so handle it
        if parameter['name'] == 'lumi':
            lumi = parameter['auxdata'][0]
            lumierr = parameter['sigmas'][0]

    # define measurement
    meas = ET.Element(
        "Measurement",
        Name=name,
        Lumi=str(lumi),
        LumiRelErr=str(lumierr),
        ExportOnly=str(True),
    )
    poiel = ET.Element('POI')
    poiel.text = poi
    meas.append(poiel)

    # add fixed parameters (constant)
    if fixed_params:
        se = ET.Element('ParamSetting', Const='True')
        se.text = ' '.join(fixed_params)
        meas.append(se)
    return meas


def build_modifier(spec, modifierspec, channelname, samplename, sampledata):
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

    attrs = {'Name': modifierspec['name']}
    if modifierspec['type'] == 'histosys':
        attrs['HistoNameLow'] = _make_hist_name(
            channelname, samplename, modifierspec['name'], suffix='Low'
        )
        attrs['HistoNameHigh'] = _make_hist_name(
            channelname, samplename, modifierspec['name'], suffix='High'
        )
        _export_root_histogram(attrs['HistoNameLow'], modifierspec['data']['lo_data'])
        _export_root_histogram(attrs['HistoNameHigh'], modifierspec['data']['hi_data'])
    elif modifierspec['type'] == 'normsys':
        attrs['High'] = str(modifierspec['data']['hi'])
        attrs['Low'] = str(modifierspec['data']['lo'])
    elif modifierspec['type'] == 'normfactor':
        # NB: only look at first measurement for normfactor configs. In order
        # to dump as HistFactory XML, this has to be the same for all
        # measurements or it will not work correctly. Why?
        #
        # Unlike other modifiers, NormFactor has the unique circumstance of
        # defining its parameter configurations at the modifier level inside
        # the channel specification, instead of at the measurement level, like
        # all of the other modifiers.
        #
        # However, since I strive for perfection, the "Const" attribute will
        # never be set here, but at the per-measurement configuration instead
        # like all other parameters. This is an acceptable compromise.
        #
        # Lastly, if a normfactor parameter configuration doesn't exist in the
        # first measurement parameter configuration, then set defaults.
        val = 1
        low = 0
        high = 10
        for p in spec['measurements'][0]['config']['parameters']:
            if p['name'] == modifierspec['name']:
                val = p.get('inits', [val])[0]
                low, high = p.get('bounds', [[low, high]])[0]
        attrs['Val'] = str(val)
        attrs['Low'] = str(low)
        attrs['High'] = str(high)
    elif modifierspec['type'] == 'staterror':
        attrs['Activate'] = 'True'
        attrs['HistoName'] = _make_hist_name(
            channelname, samplename, modifierspec['name']
        )
        # must be deleted, HiFa XML specification does not support 'Name'
        del attrs['Name']
        # need to make this a relative uncertainty stored in ROOT file
        _export_root_histogram(
            attrs['HistoName'],
            np.divide(
                modifierspec['data'],
                sampledata,
                out=np.zeros_like(sampledata),
                where=np.asarray(sampledata) != 0,
                dtype='float',
            ).tolist(),
        )
    elif modifierspec['type'] == 'shapesys':
        attrs['ConstraintType'] = 'Poisson'
        attrs['HistoName'] = _make_hist_name(
            channelname, samplename, modifierspec['name']
        )
        # need to make this a relative uncertainty stored in ROOT file
        _export_root_histogram(
            attrs['HistoName'],
            [
                np.divide(
                    a, b, out=np.zeros_like(a), where=np.asarray(b) != 0, dtype='float'
                )
                for a, b in np.array(
                    (modifierspec['data'], sampledata), dtype="float"
                ).T
            ],
        )
    elif modifierspec['type'] == 'shapefactor':
        pass
    else:
        log.warning(
            f"Skipping modifier {modifierspec['name']}({modifierspec['type']}) for now"
        )
        return None

    modifier = ET.Element(mod_map[modifierspec['type']], **attrs)
    return modifier


def build_sample(spec, samplespec, channelname):
    histname = _make_hist_name(channelname, samplespec['name'])
    attrs = {
        'Name': samplespec['name'],
        'HistoName': histname,
        'InputFile': _ROOT_DATA_FILE.file_path,
        'NormalizeByTheory': 'False',
    }
    sample = ET.Element('Sample', **attrs)
    for modspec in samplespec['modifiers']:
        # if lumi modifier added for this sample, need to set NormalizeByTheory
        if modspec['type'] == 'lumi':
            sample.attrib.update({'NormalizeByTheory': 'True'})
        modifier = build_modifier(
            spec, modspec, channelname, samplespec['name'], samplespec['data']
        )
        if modifier is not None:
            sample.append(modifier)
    _export_root_histogram(histname, samplespec['data'])
    return sample


def build_data(obsspec, channelname):
    histname = _make_hist_name(channelname, 'data')
    data = ET.Element('Data', HistoName=histname, InputFile=_ROOT_DATA_FILE.file_path)

    observation = next((obs for obs in obsspec if obs['name'] == channelname), None)
    _export_root_histogram(histname, observation['data'])
    return data


def build_channel(spec, channelspec, obsspec):
    channel = ET.Element(
        'Channel', Name=channelspec['name'], InputFile=_ROOT_DATA_FILE.file_path
    )
    if obsspec:
        data = build_data(obsspec, channelspec['name'])
        channel.append(data)
    for samplespec in channelspec['samples']:
        channel.append(build_sample(spec, samplespec, channelspec['name']))
    return channel


def writexml(spec, specdir, data_rootdir, resultprefix):
    global _ROOT_DATA_FILE

    shutil.copyfile(
        schema_path.joinpath('HistFactorySchema.dtd'),
        Path(specdir).parent.joinpath('HistFactorySchema.dtd'),
    )
    combination = ET.Element(
        "Combination", OutputFilePrefix=str(Path(specdir).joinpath(resultprefix))
    )

    with uproot.recreate(Path(data_rootdir).joinpath('data.root')) as _ROOT_DATA_FILE:
        for channelspec in spec['channels']:
            channelfilename = str(
                Path(specdir).joinpath(f'{resultprefix}_{channelspec["name"]}.xml')
            )
            with open(channelfilename, "w", encoding="utf-8") as channelfile:
                channel = build_channel(spec, channelspec, spec.get('observations'))
                indent(channel)
                channelfile.write(
                    "<!DOCTYPE Channel SYSTEM '../HistFactorySchema.dtd'>\n\n"
                )
                channelfile.write(
                    ET.tostring(channel, encoding='utf-8').decode('utf-8')
                )

            inp = ET.Element("Input")
            inp.text = channelfilename
            combination.append(inp)

    # need information about modifier types to get the right prefix in measurement
    mixin = _ChannelSummaryMixin(channels=spec['channels'])

    for measurement in spec['measurements']:
        combination.append(build_measurement(measurement, dict(mixin.modifiers)))
    indent(combination)
    return b"<!DOCTYPE Combination  SYSTEM 'HistFactorySchema.dtd'>\n\n" + ET.tostring(
        combination, encoding='utf-8'
    )

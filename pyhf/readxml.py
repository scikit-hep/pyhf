import logging

import os
import xml.etree.ElementTree as ET
import numpy as np
import tqdm
import uproot

log = logging.getLogger(__name__)

__FILECACHE__ = {}


def extract_error(h):
    """
    Determine the bin uncertainties for a histogram.

    If `fSumw2` is not filled, then the histogram must have been
    filled with no weights and `.Sumw2()` was never called. The
    bin uncertainties are then Poisson, and so the `sqrt(entries)`.

    Args:
        h (uproot.rootio.TH1 object): The histogram

    Returns:
        list: The uncertainty for each bin in the histogram
    """
    err = h.variances if h.variances.any() else h.numpy()[0]
    return np.sqrt(err).tolist()


def import_root_histogram(rootdir, filename, path, name, filecache=None):
    global __FILECACHE__
    filecache = filecache or __FILECACHE__

    # strip leading slashes as uproot doesn't use "/" for top-level
    path = path or ''
    path = path.strip('/')
    fullpath = os.path.join(rootdir, filename)
    if not fullpath in filecache:
        f = uproot.open(fullpath)
        filecache[fullpath] = f
    else:
        f = filecache[fullpath]
    try:
        h = f[name]
    except KeyError:
        try:
            h = f[os.path.join(path, name)]
        except KeyError:
            raise KeyError(
                'Both {0:s} and {1:s} were tried and not found in {2:s}'.format(
                    name, os.path.join(path, name), os.path.join(rootdir, filename)
                )
            )
    return h.numpy()[0].tolist(), extract_error(h)


def process_sample(
    sample, rootdir, inputfile, histopath, channelname, track_progress=False
):
    if 'InputFile' in sample.attrib:
        inputfile = sample.attrib.get('InputFile')
    if 'HistoPath' in sample.attrib:
        histopath = sample.attrib.get('HistoPath')
    histoname = sample.attrib['HistoName']

    data, err = import_root_histogram(rootdir, inputfile, histopath, histoname)

    modifiers = []
    # first check if we need to add lumi modifier for this sample
    if sample.attrib.get("NormalizeByTheory", "False") == 'True':
        modifiers.append({'name': 'lumi', 'type': 'lumi', 'data': None})

    modtags = tqdm.tqdm(
        sample.iter(), unit='modifier', disable=not (track_progress), total=len(sample)
    )

    for modtag in modtags:
        modtags.set_description(
            '  - modifier {0:s}({1:s})'.format(
                modtag.attrib.get('Name', 'n/a'), modtag.tag
            )
        )
        if modtag == sample:
            continue
        if modtag.tag == 'OverallSys':
            modifiers.append(
                {
                    'name': modtag.attrib['Name'],
                    'type': 'normsys',
                    'data': {
                        'lo': float(modtag.attrib['Low']),
                        'hi': float(modtag.attrib['High']),
                    },
                }
            )
        elif modtag.tag == 'NormFactor':
            modifiers.append(
                {'name': modtag.attrib['Name'], 'type': 'normfactor', 'data': None}
            )
        elif modtag.tag == 'HistoSys':
            lo, _ = import_root_histogram(
                rootdir,
                modtag.attrib.get('HistoFileLow', inputfile),
                modtag.attrib.get('HistoPathLow', ''),
                modtag.attrib['HistoNameLow'],
            )
            hi, _ = import_root_histogram(
                rootdir,
                modtag.attrib.get('HistoFileHigh', inputfile),
                modtag.attrib.get('HistoPathHigh', ''),
                modtag.attrib['HistoNameHigh'],
            )
            modifiers.append(
                {
                    'name': modtag.attrib['Name'],
                    'type': 'histosys',
                    'data': {'lo_data': lo, 'hi_data': hi},
                }
            )
        elif modtag.tag == 'StatError' and modtag.attrib['Activate'] == 'True':
            if modtag.attrib.get('HistoName', '') == '':
                staterr = err
            else:
                extstat, _ = import_root_histogram(
                    rootdir,
                    modtag.attrib.get('HistoFile', inputfile),
                    modtag.attrib.get('HistoPath', ''),
                    modtag.attrib['HistoName'],
                )
                staterr = np.multiply(extstat, data).tolist()
            if not staterr:
                raise RuntimeError('cannot determine stat error.')
            modifiers.append(
                {
                    'name': 'staterror_{}'.format(channelname),
                    'type': 'staterror',
                    'data': staterr,
                }
            )
        elif modtag.tag == 'ShapeSys':
            # NB: ConstraintType is ignored
            if modtag.attrib.get('ConstraintType', 'Poisson') != 'Poisson':
                log.warning(
                    'shapesys modifier %s has a non-poisson constraint',
                    modtag.attrib['Name'],
                )
            shapesys_data, _ = import_root_histogram(
                rootdir,
                modtag.attrib.get('InputFile', inputfile),
                modtag.attrib.get('HistoPath', ''),
                modtag.attrib['HistoName'],
            )
            # NB: we convert relative uncertainty to absolute uncertainty
            modifiers.append(
                {
                    'name': modtag.attrib['Name'],
                    'type': 'shapesys',
                    'data': [a * b for a, b in zip(data, shapesys_data)],
                }
            )
        else:
            log.warning('not considering modifier tag %s', modtag)

    return {'name': sample.attrib['Name'], 'data': data, 'modifiers': modifiers}


def process_data(sample, rootdir, inputfile, histopath):
    if 'InputFile' in sample.attrib:
        inputfile = sample.attrib.get('InputFile')
    if 'HistoPath' in sample.attrib:
        histopath = sample.attrib.get('HistoPath')
    histoname = sample.attrib['HistoName']

    data, _ = import_root_histogram(rootdir, inputfile, histopath, histoname)
    return data


def process_channel(channelxml, rootdir, track_progress=False):
    channel = channelxml.getroot()

    inputfile = channel.attrib.get('InputFile')
    histopath = channel.attrib.get('HistoPath')

    samples = tqdm.tqdm(
        channel.findall('Sample'), unit='sample', disable=not (track_progress)
    )

    data = channel.findall('Data')
    if data:
        parsed_data = process_data(data[0], rootdir, inputfile, histopath)
    else:
        parsed_data = None
    channelname = channel.attrib['Name']

    results = []
    for sample in samples:
        samples.set_description('  - sample {}'.format(sample.attrib.get('Name')))
        result = process_sample(
            sample, rootdir, inputfile, histopath, channelname, track_progress
        )
        results.append(result)

    return channelname, parsed_data, results


def process_measurements(toplvl):
    results = []
    for x in toplvl.findall('Measurement'):
        lumi = float(x.attrib['Lumi'])
        lumierr = lumi * float(x.attrib['LumiRelErr'])

        result = {
            'name': x.attrib['Name'],
            'config': {
                'poi': x.findall('POI')[0].text,
                'parameters': [
                    {
                        'name': 'lumi',
                        'auxdata': [lumi],
                        'bounds': [[lumi - 5.0 * lumierr, lumi + 5.0 * lumierr]],
                        'inits': [lumi],
                        'sigmas': [lumierr],
                    }
                ],
            },
        }
        for param in x.findall('ParamSetting'):
            # determine what all parameters in the paramsetting have in common
            overall_param_obj = {}
            if param.attrib.get('Const'):
                overall_param_obj['fixed'] = param.attrib['Const'] == 'True'
            if param.attrib.get('Val'):
                overall_param_obj['value'] = param.attrib['Val']

            # might be specifying multiple parameters in the same ParamSetting
            if param.text:
                for param_name in param.text.split(' '):
                    # lumi will always be the first parameter
                    if param_name == 'Lumi':
                        result['config']['parameters'][0].update(overall_param_obj)
                    else:
                        param_obj = {'name': param_name}
                        param_obj.update(overall_param_obj)
                        result['config']['parameters'].append(param_obj)
        results.append(result)
    return results


def parse(configfile, rootdir, track_progress=False):
    toplvl = ET.parse(configfile)
    inputs = tqdm.tqdm(
        [x.text for x in toplvl.findall('Input')],
        unit='channel',
        disable=not (track_progress),
    )

    channels = {}
    for inp in inputs:
        inputs.set_description('Processing {}'.format(inp))
        channel, data, samples = process_channel(
            ET.parse(os.path.join(rootdir, inp)), rootdir, track_progress
        )
        channels[channel] = {'data': data, 'samples': samples}

    return {
        'toplvl': {
            'resultprefix': toplvl.getroot().attrib['OutputFilePrefix'],
            'measurements': process_measurements(toplvl),
        },
        'channels': [{'name': k, 'samples': v['samples']} for k, v in channels.items()],
        'data': {k: v['data'] for k, v in channels.items()},
    }


def clear_filecache():
    global __FILECACHE__
    __FILECACHE__ = {}

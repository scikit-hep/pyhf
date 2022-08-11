from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Callable, Iterable, Tuple, Union, IO
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import tqdm
import uproot

from pyhf import compat
from pyhf import exceptions
from pyhf import schema

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    PathOrStr = Union[str, os.PathLike[str]]
else:
    PathOrStr = Union[str, "os.PathLike[str]"]

__FILECACHE__ = {}
MountPathType = Iterable[Tuple[Path, Path]]

__all__ = [
    "clear_filecache",
    "dedupe_parameters",
    "extract_error",
    "import_root_histogram",
    "parse",
    "process_channel",
    "process_data",
    "process_measurements",
    "process_sample",
]


def __dir__():
    return __all__


def resolver_factory(rootdir: Path, mounts: MountPathType) -> Callable[[str], Path]:
    def resolver(filename: str) -> Path:
        path = Path(filename)
        for host_path, mount_path in mounts:
            # NB: path.parents doesn't include the path itself, which might be
            # a directory as well, so check that edge case
            if mount_path == path or mount_path in path.parents:
                path = host_path.joinpath(path.relative_to(mount_path))
                break
        return rootdir.joinpath(path)

    return resolver


def extract_error(hist):
    """
    Determine the bin uncertainties for a histogram.

    If `fSumw2` is not filled, then the histogram must have been
    filled with no weights and `.Sumw2()` was never called. The
    bin uncertainties are then Poisson, and so the `sqrt(entries)`.

    Args:
        hist (:class:`uproot.behaviors.TH1.TH1`): The histogram

    Returns:
        list: The uncertainty for each bin in the histogram
    """

    variance = hist.variances() if hist.weighted else hist.to_numpy()[0]
    return np.sqrt(variance).tolist()


def import_root_histogram(resolver, filename, path, name, filecache=None):
    global __FILECACHE__
    filecache = filecache or __FILECACHE__

    # strip leading slashes as uproot doesn't use "/" for top-level
    path = path or ''
    path = path.strip('/')
    fullpath = str(resolver(filename))
    if fullpath not in filecache:
        f = uproot.open(fullpath)
        keys = set(f.keys(cycle=False))
        filecache[fullpath] = (f, keys)
    else:
        f, keys = filecache[fullpath]

    fullname = "/".join([path, name])

    if name in keys:
        hist = f[name]
    elif fullname in keys:
        hist = f[fullname]
    else:
        raise KeyError(
            f'Both {name} and {fullname} were tried and not found in {fullpath}'
        )
    return hist.to_numpy()[0].tolist(), extract_error(hist)


def process_sample(
    sample, resolver, inputfile, histopath, channel_name, track_progress=False
):
    if 'InputFile' in sample.attrib:
        inputfile = sample.attrib.get('InputFile')
    if 'HistoPath' in sample.attrib:
        histopath = sample.attrib.get('HistoPath')
    histoname = sample.attrib['HistoName']

    data, err = import_root_histogram(resolver, inputfile, histopath, histoname)

    parameter_configs = []
    modifiers = []
    # first check if we need to add lumi modifier for this sample
    if sample.attrib.get("NormalizeByTheory", "False") == 'True':
        modifiers.append({'name': 'lumi', 'type': 'lumi', 'data': None})

    modtags = tqdm.tqdm(
        sample.iter(), unit='modifier', disable=not (track_progress), total=len(sample)
    )

    for modtag in modtags:
        modtags.set_description(
            f"  - modifier {modtag.attrib.get('Name', 'n/a'):s}({modtag.tag:s})"
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
            parameter_config = {
                'name': modtag.attrib['Name'],
                'bounds': [[float(modtag.attrib['Low']), float(modtag.attrib['High'])]],
                'inits': [float(modtag.attrib['Val'])],
            }
            if modtag.attrib.get('Const'):
                parameter_config['fixed'] = modtag.attrib['Const'] == 'True'

            parameter_configs.append(parameter_config)
        elif modtag.tag == 'HistoSys':
            lo, _ = import_root_histogram(
                resolver,
                modtag.attrib.get('HistoFileLow', inputfile),
                modtag.attrib.get('HistoPathLow', ''),
                modtag.attrib['HistoNameLow'],
            )
            hi, _ = import_root_histogram(
                resolver,
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
                    resolver,
                    modtag.attrib.get('HistoFile', inputfile),
                    modtag.attrib.get('HistoPath', ''),
                    modtag.attrib['HistoName'],
                )
                staterr = np.multiply(extstat, data).tolist()
            if not staterr:
                raise RuntimeError('cannot determine stat error.')
            modifiers.append(
                {
                    'name': f'staterror_{channel_name}',
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
                resolver,
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
        elif modtag.tag == 'ShapeFactor':
            modifiers.append(
                {'name': modtag.attrib['Name'], 'type': 'shapefactor', 'data': None}
            )
        else:
            log.warning('not considering modifier tag %s', modtag)

    return {
        'name': sample.attrib['Name'],
        'data': data,
        'modifiers': modifiers,
        'parameter_configs': parameter_configs,
    }


def process_data(sample, resolver, inputfile, histopath):
    if 'InputFile' in sample.attrib:
        inputfile = sample.attrib.get('InputFile')
    if 'HistoPath' in sample.attrib:
        histopath = sample.attrib.get('HistoPath')
    histoname = sample.attrib['HistoName']

    data, _ = import_root_histogram(resolver, inputfile, histopath, histoname)
    return data


def process_channel(channelxml, resolver, track_progress=False):
    channel = channelxml.getroot()

    inputfile = channel.attrib.get('InputFile')
    histopath = channel.attrib.get('HistoPath')

    samples = tqdm.tqdm(
        channel.findall('Sample'), unit='sample', disable=not (track_progress)
    )

    channel_name = channel.attrib['Name']

    data = channel.findall('Data')
    if data:
        parsed_data = process_data(data[0], resolver, inputfile, histopath)
    else:
        raise RuntimeError(f"Channel {channel_name} is missing data. See issue #1911.")

    results = []
    channel_parameter_configs = []
    for sample in samples:
        samples.set_description(f"  - sample {sample.attrib.get('Name')}")
        result = process_sample(
            sample, resolver, inputfile, histopath, channel_name, track_progress
        )
        channel_parameter_configs.extend(result.pop('parameter_configs'))
        results.append(result)

    return channel_name, parsed_data, results, channel_parameter_configs


def process_measurements(toplvl, other_parameter_configs=None):
    """
    For a given XML structure, provide a parsed dictionary adhering to defs.json/#definitions/measurement.

    Args:
        toplvl (:mod:`xml.etree.ElementTree`): The top-level XML document to parse.
        other_parameter_configs (:obj:`list`): A list of other parameter configurations from other non-top-level XML documents to incorporate into the resulting measurement object.

    Returns:
        :obj:`dict`: A measurement object.

    """
    results = []
    other_parameter_configs = other_parameter_configs if other_parameter_configs else []

    for x in toplvl.findall('Measurement'):
        parameter_configs_map = {k['name']: dict(**k) for k in other_parameter_configs}
        lumi = float(x.attrib['Lumi'])
        lumierr = lumi * float(x.attrib['LumiRelErr'])

        measurement_name = x.attrib['Name']

        poi = x.find('POI')
        if poi is None:
            raise RuntimeError(
                f"Measurement {measurement_name} is missing POI specification"
            )

        result = {
            'name': measurement_name,
            'config': {
                'poi': poi.text.strip() if poi.text else '',
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
                overall_param_obj['inits'] = [float(param.attrib['Val'])]

            # might be specifying multiple parameters in the same ParamSetting
            if param.text:
                for param_name in param.text.strip().split(' '):
                    param_interpretation = compat.interpret_rootname(param_name)
                    if not param_interpretation['is_scalar']:
                        raise ValueError(
                            f'pyhf does not support setting non-scalar parameters ("gammas")  constant, such as for {param_name}.'
                        )
                    if param_interpretation['name'] == 'lumi':
                        result['config']['parameters'][0].update(overall_param_obj)
                    else:
                        # pop from parameter_configs_map because we don't want to duplicate
                        param_obj = parameter_configs_map.pop(
                            param_interpretation['name'],
                            {'name': param_interpretation['name']},
                        )
                        # ParamSetting will always take precedence
                        param_obj.update(overall_param_obj)
                        # add it back in to the parameter_configs_map
                        parameter_configs_map[param_interpretation['name']] = param_obj
        result['config']['parameters'].extend(parameter_configs_map.values())
        results.append(result)

    return results


def dedupe_parameters(parameters):
    duplicates = {}
    for p in parameters:
        duplicates.setdefault(p['name'], []).append(p)
    for parname in duplicates:
        parameter_list = duplicates[parname]
        if len(parameter_list) == 1:
            continue
        elif any(p != parameter_list[0] for p in parameter_list[1:]):
            for p in parameter_list:
                log.warning(p)
            raise RuntimeError(
                f'cannot import workspace due to incompatible parameter configurations for {parname:s}.'
            )
    # no errors raised, de-dupe and return
    return list({v['name']: v for v in parameters}.values())


def parse(
    configfile: PathOrStr | IO[bytes] | IO[str],
    rootdir: PathOrStr,
    mounts: MountPathType | None = None,
    track_progress: bool = False,
    validation_as_error: bool = True,
):
    """
    Parse the ``configfile`` with respect to the ``rootdir``.

    Args:
        configfile (:class:`pathlib.Path` or :obj:`str` or file object): The top-level XML config file to parse.
        rootdir (:class:`pathlib.Path` or :obj:`str`): The path to the working directory for interpreting relative paths in the configuration.
        mounts (:obj:`None` or :obj:`list` of 2-:obj:`tuple` of :class:`pathlib.Path` objects): The first field is the local path to where files are located, the second field is the path where the file or directory are saved in the XML configuration. This is similar in spirit to Docker volume mounts. Default is ``None``.
        track_progress (:obj:`bool`): Show the progress bar. Default is to hide the progress bar.
        validation_as_error (:obj:`bool`): Throw an exception (``True``) or print a warning (``False``) if the resulting HistFactory JSON does not adhere to the schema. Default is to throw an exception.

    Returns:
        spec (:obj:`jsonable`): The newly built HistFactory JSON specification
    """
    mounts = mounts or []
    toplvl = ET.parse(configfile)
    inputs = tqdm.tqdm(
        [x.text for x in toplvl.findall('Input')],
        unit='channel',
        disable=not (track_progress),
    )

    # create a resolver for finding files
    resolver = resolver_factory(Path(rootdir), mounts)

    channels = {}
    parameter_configs = []
    for inp in inputs:
        inputs.set_description(f'Processing {inp}')
        channel, data, samples, channel_parameter_configs = process_channel(
            ET.parse(resolver(inp)), resolver, track_progress
        )
        channels[channel] = {'data': data, 'samples': samples}
        parameter_configs.extend(channel_parameter_configs)

    parameter_configs = dedupe_parameters(parameter_configs)
    result = {
        'measurements': process_measurements(
            toplvl, other_parameter_configs=parameter_configs
        ),
        'channels': [
            {'name': channel_name, 'samples': channel_spec['samples']}
            for channel_name, channel_spec in channels.items()
        ],
        'observations': [
            {'name': channel_name, 'data': channel_spec['data']}
            for channel_name, channel_spec in channels.items()
        ],
        'version': schema.version,
    }
    try:
        schema.validate(result, 'workspace.json')
    except exceptions.InvalidSpecification as exc:
        if validation_as_error:
            raise exc
        else:
            log.warning(exc)
    return result


def clear_filecache():
    global __FILECACHE__
    __FILECACHE__ = {}

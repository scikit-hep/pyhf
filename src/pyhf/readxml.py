from __future__ import annotations

import logging
from typing import (
    IO,
    Callable,
    Union,
    cast,
)
from collections.abc import Iterable, MutableMapping, MutableSequence, Sequence

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
)
from rich.console import Console
import uproot

from pyhf import compat
from pyhf import exceptions
from pyhf import schema
from pyhf.typing import (
    Channel,
    HistoSys,
    LumiSys,
    Measurement,
    Modifier,
    NormFactor,
    NormSys,
    Observation,
    Parameter,
    ParameterBase,
    PathOrStr,
    Sample,
    ShapeFactor,
    ShapeSys,
    StatError,
    Workspace,
)

log = logging.getLogger(__name__)

FileCacheType = MutableMapping[str, tuple[Union[IO[str], IO[bytes]], set[str]]]
MountPathType = Iterable[tuple[Path, Path]]
ResolverType = Callable[[str], Path]

__FILECACHE__: FileCacheType = {}

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


def __dir__() -> list[str]:
    return __all__


def resolver_factory(rootdir: Path, mounts: MountPathType) -> ResolverType:
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


def extract_error(hist: uproot.behaviors.TH1.TH1) -> list[float]:
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
    return cast(list[float], np.sqrt(variance).tolist())


def import_root_histogram(
    resolver: ResolverType,
    filename: str,
    path: str,
    name: str,
    filecache: FileCacheType | None = None,
) -> tuple[list[float], list[float]]:
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
    sample: ET.Element,
    resolver: ResolverType,
    inputfile: str,
    histopath: str,
    channel_name: str,
    track_progress: bool = False,
    progress: Progress | None = None,
) -> Sample:
    inputfile = sample.attrib.get('InputFile', inputfile)
    histopath = sample.attrib.get('HistoPath', histopath)
    histoname = sample.attrib['HistoName']

    data, err = import_root_histogram(resolver, inputfile, histopath, histoname)

    parameter_configs: MutableSequence[Parameter] = []
    modifiers: MutableSequence[Modifier] = []
    # first check if we need to add lumi modifier for this sample
    if sample.attrib.get("NormalizeByTheory", "False") == 'True':
        modifier_lumi: LumiSys = {'name': 'lumi', 'type': 'lumi', 'data': None}
        modifiers.append(modifier_lumi)

    samples = list(sample.iter())

    if progress is not None and track_progress:
        task_id = progress.add_task("[cyan]  - processing modifiers", total=len(sample))

    for modtag in samples:
        if track_progress and progress is not None:
            progress.update(
                task_id,
                description=f"[cyan]  - modifier {modtag.attrib.get('Name', 'n/a'):s}({modtag.tag:s})",
            )

        if modtag == sample:
            continue
        if modtag.tag == 'OverallSys':
            modifier_normsys: NormSys = {
                'name': modtag.attrib['Name'],
                'type': 'normsys',
                'data': {
                    'lo': float(modtag.attrib['Low']),
                    'hi': float(modtag.attrib['High']),
                },
            }
            modifiers.append(modifier_normsys)
        elif modtag.tag == 'NormFactor':
            modifier_normfactor: NormFactor = {
                'name': modtag.attrib['Name'],
                'type': 'normfactor',
                'data': None,
            }
            modifiers.append(modifier_normfactor)
            parameter_config: Parameter = {
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
            modifier_histosys: HistoSys = {
                'name': modtag.attrib['Name'],
                'type': 'histosys',
                'data': {'lo_data': lo, 'hi_data': hi},
            }
            modifiers.append(modifier_histosys)
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
            modifier_staterror: StatError = {
                'name': f'staterror_{channel_name}',
                'type': 'staterror',
                'data': staterr,
            }
            modifiers.append(modifier_staterror)
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
            modifier_shapesys: ShapeSys = {
                'name': modtag.attrib['Name'],
                'type': 'shapesys',
                'data': [a * b for a, b in zip(data, shapesys_data)],
            }
            modifiers.append(modifier_shapesys)
        elif modtag.tag == 'ShapeFactor':
            modifier_shapefactor: ShapeFactor = {
                'name': modtag.attrib['Name'],
                'type': 'shapefactor',
                'data': None,
            }
            modifiers.append(modifier_shapefactor)
        else:
            log.warning('not considering modifier tag %s', modtag)

        if track_progress and progress is not None:
            progress.advance(task_id, 1)

    if track_progress and progress is not None:
        progress.remove_task(task_id)

    return {
        'name': sample.attrib['Name'],
        'data': data,
        'modifiers': modifiers,
        'parameter_configs': parameter_configs,
    }


def process_data(
    sample: ET.Element,
    resolver: ResolverType,
    inputfile: str,
    histopath: str,
) -> list[float]:
    inputfile = sample.attrib.get('InputFile', inputfile)
    histopath = sample.attrib.get('HistoPath', histopath)
    histoname = sample.attrib['HistoName']

    if inputfile == "" or histoname == "":
        raise NotImplementedError(
            "Conversion of workspaces without data is currently not supported.\nSee https://github.com/scikit-hep/pyhf/issues/566"
        )

    data, _ = import_root_histogram(resolver, inputfile, histopath, histoname)
    return data


def process_channel(
    channelxml: ET.ElementTree[ET.Element[str]],
    resolver: ResolverType,
    track_progress: bool = False,
    progress: Progress | None = None,
) -> tuple[str, list[float], list[Sample], list[Parameter]]:
    channel = channelxml.getroot()
    if channel is None:
        raise RuntimeError("Root element of ElementTree is missing.")

    inputfile = channel.attrib.get('InputFile', '')
    histopath = channel.attrib.get('HistoPath', '')

    samples = channel.findall("Sample")

    channel_name = channel.attrib['Name']

    data = channel.findall('Data')
    if data:
        parsed_data = process_data(data[0], resolver, inputfile, histopath)
    else:
        raise RuntimeError(f"Channel {channel_name} is missing data. See issue #1911.")

    if progress is not None and track_progress:
        task_id = progress.add_task("[cyan]  - processing samples", total=len(samples))

    results = []
    channel_parameter_configs: list[Parameter] = []
    for sample in samples:
        if track_progress and progress is not None:
            progress.update(
                task_id, description=f"[cyan]  - sample {sample.attrib.get('Name')}"
            )

        result = process_sample(
            sample,
            resolver,
            inputfile,
            histopath,
            channel_name,
            track_progress,
            progress,
        )
        channel_parameter_configs.extend(result.pop('parameter_configs'))
        results.append(result)

        if track_progress and progress is not None:
            progress.advance(task_id, 1)

    if track_progress and progress is not None:
        progress.remove_task(task_id)

    return channel_name, parsed_data, results, channel_parameter_configs


def process_measurements(
    toplvl: ET.ElementTree[ET.Element[str]],
    other_parameter_configs: Sequence[Parameter] | None = None,
) -> list[Measurement]:
    """
    For a given XML structure, provide a parsed dictionary adhering to defs.json/#definitions/measurement.

    Args:
        toplvl (:mod:`xml.etree.ElementTree`): The top-level XML document to parse.
        other_parameter_configs (:obj:`list`): A list of other parameter configurations from other non-top-level XML documents to incorporate into the resulting measurement object.

    Returns:
        :obj:`dict`: A measurement object.

    """
    results: list[Measurement] = []
    other_parameter_configs = other_parameter_configs if other_parameter_configs else []

    for x in toplvl.findall('Measurement'):
        parameter_configs_map: MutableMapping[str, Parameter] = {
            k['name']: dict(**k) for k in other_parameter_configs
        }
        lumi = float(x.attrib['Lumi'])
        lumierr = lumi * float(x.attrib['LumiRelErr'])

        measurement_name = x.attrib['Name']

        poi = x.find('POI')
        if poi is None:
            raise RuntimeError(
                f"Measurement {measurement_name} is missing POI specification"
            )

        result: Measurement = {
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
            overall_param_obj: ParameterBase = {}
            if param.attrib.get('Const'):
                overall_param_obj['fixed'] = param.attrib['Const'] == 'True'
            if param.attrib.get('Val'):
                overall_param_obj['inits'] = [float(param.attrib['Val'])]

            # might be specifying multiple parameters in the same ParamSetting
            if param.text:
                for param_name in param.text.strip().split(' '):
                    param_interpretation = compat.interpret_rootname(param_name)  # type: ignore[no-untyped-call]
                    if not param_interpretation['is_scalar']:
                        raise ValueError(
                            f'pyhf does not support setting non-scalar parameters ("gammas")  constant, such as for {param_name}.'
                        )
                    if param_interpretation['name'] == 'lumi':
                        result['config']['parameters'][0].update(overall_param_obj)  # type: ignore[typeddict-item]
                    else:
                        # pop from parameter_configs_map because we don't want to duplicate
                        param_obj: Parameter = parameter_configs_map.pop(
                            param_interpretation['name'],
                            {'name': param_interpretation['name']},
                        )
                        # ParamSetting will always take precedence
                        param_obj.update(overall_param_obj)  # type: ignore[typeddict-item]
                        # add it back in to the parameter_configs_map
                        parameter_configs_map[param_interpretation['name']] = param_obj
        result['config']['parameters'].extend(parameter_configs_map.values())
        results.append(result)

    return results


def dedupe_parameters(parameters: Sequence[Parameter]) -> list[Parameter]:
    duplicates: MutableMapping[str, MutableSequence[Parameter]] = {}
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
) -> Workspace:
    """
    Parse the ``configfile`` with respect to the ``rootdir``.

    Args:
        configfile (:class:`pathlib.Path` or :obj:`str` or file object): The top-level XML config file to parse.
        rootdir (:class:`pathlib.Path` or :obj:`str`): The path to the working directory for interpreting relative paths in the configuration.
        mounts (:obj:`None` or :obj:`list` of 2-:obj:`tuple` of :class:`os.PathLike` objects): The first field is the local path to where files are located, the second field is the path where the file or directory are saved in the XML configuration. This is similar in spirit to Docker volume mounts. Default is ``None``.
        track_progress (:obj:`bool`): Show the progress bar. Default is to hide the progress bar.
        validation_as_error (:obj:`bool`): Throw an exception (``True``) or print a warning (``False``) if the resulting HistFactory JSON does not adhere to the schema. Default is to throw an exception.

    Returns:
        spec (:obj:`jsonable`): The newly built HistFactory JSON specification
    """
    mounts = mounts or []
    toplvl = ET.parse(configfile)
    inputs = [x.text for x in toplvl.findall("Input") if x.text]

    # create a resolver for finding files
    resolver = resolver_factory(Path(rootdir), mounts)

    channels: MutableSequence[Channel] = []
    observations: MutableSequence[Observation] = []
    parameter_configs = []

    if track_progress:
        console = Console(stderr=True)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task(
                "[bold blue]Processing channels", total=len(inputs)
            )

            for input in inputs:
                progress.update(task_id, description=f"[bold blue]Processing {input}")
                channel, data, samples, channel_parameter_configs = process_channel(
                    ET.parse(resolver(input)), resolver, track_progress, progress
                )
                channels.append({"name": channel, "samples": samples})
                observations.append({"name": channel, "data": data})
                parameter_configs.extend(channel_parameter_configs)
                progress.advance(task_id, 1)
    else:
        # No progress tracking
        for input in inputs:
            channel, data, samples, channel_parameter_configs = process_channel(
                ET.parse(resolver(input)), resolver, track_progress, None
            )
            channels.append({"name": channel, "samples": samples})
            observations.append({"name": channel, "data": data})
            parameter_configs.extend(channel_parameter_configs)

    parameter_configs = dedupe_parameters(parameter_configs)
    measurements = process_measurements(
        toplvl, other_parameter_configs=parameter_configs
    )
    result: Workspace = {
        'measurements': measurements,
        'channels': channels,
        'observations': observations,
        'version': schema.version,  # type: ignore[typeddict-unknown-key]
    }
    try:
        schema.validate(result, 'workspace.json')
    except exceptions.InvalidSpecification as exc:
        if validation_as_error:
            raise exc
        else:
            log.warning(exc)
    return result


def clear_filecache() -> None:
    global __FILECACHE__
    __FILECACHE__ = {}

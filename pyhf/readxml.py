import os
import xml.etree.ElementTree as ET
import numpy as np
import logging

log = logging.getLogger(__name__)

def import_root_histogram(rootdir, filename, path, name):
    import uproot
    #import pdb; pdb.set_trace()
    #assert path == ''
    # strip leading slashes as uproot doesn't use "/" for top-level
    path = path.strip('/')
    f = uproot.open(os.path.join(rootdir, filename))
    try:
        h = f[name]
        return h.numpy[0].tolist(), np.sqrt(h.fSumw2[1:-1]).tolist()
    except KeyError:
        pass

    try:
        h = f[os.path.join(path, name)]
        return h.numpy[0].tolist(), np.sqrt(h.fSumw2[1:-1]).tolist()
    except KeyError:
        pass

    raise KeyError('Both {0:s} and {1:s} were tried and not found in {2:s}'.format(name, os.path.join(path, name), os.path.join(rootdir, filename)))

def process_sample(sample,rootdir,inputfile, histopath, channelname):
    if 'InputFile' in sample.attrib:
       inputfile = sample.attrib.get('InputFile')
    if 'HistoPath' in sample.attrib:
        histopath = sample.attrib.get('HistoPath')
    histoname = sample.attrib['HistoName']

    data,err = import_root_histogram(rootdir, inputfile, histopath, histoname)

    modifiers = []
    for modtag in sample.iter():
        if modtag == sample:
            continue
        if modtag.tag == 'OverallSys':
            modifiers.append({
                'name': modtag.attrib['Name'],
                'type': 'normsys',
                'data': {'lo': float(modtag.attrib['Low']), 'hi': float(modtag.attrib['High'])}
            })
        elif modtag.tag == 'NormFactor':
            modifiers.append({
                'name': modtag.attrib['Name'],
                'type': 'normfactor',
                'data': None
            })

        elif modtag.tag == 'HistoSys':
            lo,_ = import_root_histogram(rootdir,
                    modtag.attrib.get('HistoFileLow',inputfile),
                    modtag.attrib.get('HistoPathLow',''),
                    modtag.attrib['HistoNameLow']
                    )
            hi,_ = import_root_histogram(rootdir,
                    modtag.attrib.get('HistoFileHigh',inputfile),
                    modtag.attrib.get('HistoPathHigh',''),
                    modtag.attrib['HistoNameHigh']
                    )
            modifiers.append({
                'name': modtag.attrib['Name'],
                'type': 'histosys',
                'data': {'lo_data': lo, 'hi_data': hi}
            })
        elif modtag.tag == 'StatError' and modtag.attrib['Activate'] == 'True':
            if modtag.attrib.get('HistoName','') == '':
                modifiers.append({
                    'name': 'staterror_{}'.format(channelname),
                    'type': 'staterror',
                    'data': err
                })
            else:
                log.warning('not considering stat error with external uncrt %s', sample.attrib['Name'])
        else:
            log.warning('not considering modifier tag %s', modtag)


    return {
        'name': sample.attrib['Name'],
        'data': data,
        'modifiers': modifiers
    }

def process_data(sample,rootdir,inputfile, histopath):
    if 'InputFile' in sample.attrib:
       inputfile = sample.attrib.get('InputFile')
    if 'HistoPath' in sample.attrib:
        histopath = sample.attrib.get('HistoPath')
    histoname = sample.attrib['HistoName']

    data,_ = import_root_histogram(rootdir, inputfile, histopath, histoname)
    return data

def process_channel(channelxml,rootdir):
    channel = channelxml.getroot()

    inputfile = channel.attrib.get('InputFile')
    histopath = channel.attrib.get('HistoPath')

    samples = channel.findall('Sample')


    data = channel.findall('Data')[0]

    channelname = channel.attrib['Name']
    return  channelname, process_data(data, rootdir, inputfile, histopath), [process_sample(x, rootdir, inputfile, histopath, channelname) for x in samples]

def parse(configfile,rootdir):
    toplvl = ET.parse(configfile)
    inputs = [ET.parse(os.path.join(rootdir,x.text)) for x in toplvl.findall('Input')]
    channels = {
        k:{'data': d, 'samples': v} for k,d,v in [process_channel(inp,rootdir) for inp in inputs]
    }
    return {
        'toplvl':{
            'resultprefix':toplvl.getroot().attrib['OutputFilePrefix'],
            'measurements': {x.attrib['Name'] : {} for x in toplvl.findall('Measurement')}
        },
        'channels': [{'name': k, 'samples': v['samples']} for k,v in channels.items()],
        'data':     {k:v['data'] for k,v in channels.items()}
    }

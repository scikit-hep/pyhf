import os
import xml.etree.cElementTree as ET

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


def build_measurement(measurement):
    config = measurement['config']
    name = measurement['name']
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


def write_channel(channelspec, filename, data_rootdir):
    # need to write channelfile here
    with open(filename, 'w') as f:
        channel = ET.Element('Channel', Name=channelspec['name'])
        channel = ET.Element('Channel', Name=channelspec['name'])
        f.write(ET.tostring(channel, encoding='utf-8').decode('utf-8'))
    pass


def writexml(spec, specdir, data_rootdir, result_outputprefix):
    combination = ET.Element("Combination", OutputFilePrefix=result_outputprefix)

    for c in spec['channels']:
        channelfilename = os.path.join(specdir, 'channel_{}.xml'.format(c['name']))
        write_channel(c, channelfilename, data_rootdir)
        inp = ET.Element("Input")
        inp.text = channelfilename
        combination.append(inp)

    for measurement in spec['toplvl']['measurements']:
        combination.append(build_measurement(measurement))
    indent(combination)
    return ET.tostring(combination, encoding='utf-8')

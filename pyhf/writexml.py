import os
import xml.etree.cElementTree as ET

def measurement(lumi, lumierr, poi, param_settings = None, name = 'Meas1'):
    param_settings = param_settings or []

    meas = ET.Element("Measurement", Name = name, Lumi = str(lumi), LumiRelErr = str(lumierr))
    poiel  = ET.Element('POI')
    poiel.text = poi
    meas.append(poiel)
    for s in param_settings:
        se  = ET.Element('ParamSetting', **s['attrs'])
        se.text = ' '.join(s['params'])
        meas.append(se)
    return meas

def write_channel(channelspec, filename, data_rootdir):
    #need to write channelfile here
    with open(filename,'w') as f:
        channel = ET.Element('Channel', Name = channelspec['name'])
        channel = ET.Element('Channel', Name = channelspec['name'])
        f.write(ET.tostring(channel, encoding = 'utf-8').decode('utf-8'))
    pass


def writexml(spec, specdir, data_rootdir , result_outputprefix):
    combination = ET.Element("Combination", OutputFilePrefix = result_outputprefix)

    for c in spec['channels']:
        channelfilename = os.path.join(specdir,'channel_{}.xml'.format(c['name']))
        write_channel(c,channelfilename,data_rootdir)
        inp = ET.Element("Input")        
        inp.text = channelfilename
        combination.append(inp)


    m = measurement(1,0.1,'SigXsecOverSM',[{'attrs': {'Const': 'True'}, 'params': ['Lumi' 'alpha_syst1']}])
    combination.append(m)
    return ET.tostring(combination, encoding = 'utf-8')


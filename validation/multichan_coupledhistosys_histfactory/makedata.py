import ROOT

import json
import sys

source_data = json.load(open(sys.argv[1]))
root_file = sys.argv[2]

f = ROOT.TFile(root_file, 'RECREATE')


hists = []
for cname, channel_def in source_data['channels'].iteritems():
    print('CH', cname)
    binning = channel_def['binning']
    bindata = channel_def['bindata']

    for hist, data in bindata.iteritems():
        print(f'{cname}_{hist}')
        h = ROOT.TH1F(f'{cname}_{hist}', f'{cname}_{hist}', *binning)
        hists += [h]
        for i, v in enumerate(data):
            h.SetBinContent(i + 1, v)
        h.Sumw2()

f.Write()

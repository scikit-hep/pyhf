import ROOT

import json
import sys

with open(sys.argv[1], encoding="utf-8") as source_file:
    source_data = json.load(source_file)
root_file = sys.argv[2]

f = ROOT.TFile(root_file, 'RECREATE')

for cname, channel_def in source_data['channels'].iteritems():
    print('CH', cname)
    binning = channel_def['binning']
    bindata = channel_def['bindata']

    data = ROOT.TH1F(f'{cname}_data', f'{cname}_data', *binning)
    for i, v in enumerate(bindata['data']):
        data.SetBinContent(i + 1, v)
    data.Sumw2()

    print(data.GetName())

    bkg = ROOT.TH1F(f'{cname}_bkg', f'{cname}_bkg', *binning)
    for i, v in enumerate(bindata['bkg']):
        bkg.SetBinContent(i + 1, v)
    bkg.Sumw2()

    if 'bkgerr' in bindata:
        bkgerr = ROOT.TH1F(f'{cname}_bkgerr', f'{cname}_bkgerr', *binning)

        # shapesys must be as multiplicative factor
        for i, v in enumerate(bindata['bkgerr']):
            bkgerr.SetBinContent(i + 1, v / bkg.GetBinContent(i + 1))
        bkgerr.Sumw2()

    if 'sig' in bindata:
        sig = ROOT.TH1F(f'{cname}_signal', f'{cname}_signal', *binning)
        for i, v in enumerate(bindata['sig']):
            sig.SetBinContent(i + 1, v)
        sig.Sumw2()
    f.Write()

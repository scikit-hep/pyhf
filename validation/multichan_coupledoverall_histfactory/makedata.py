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

    bkg1 = ROOT.TH1F(f'{cname}_bkg1', f'{cname}_bkg1', *binning)
    for i, v in enumerate(bindata['bkg1']):
        bkg1.SetBinContent(i + 1, v)
    bkg1.Sumw2()

    if 'bkg2' in bindata:
        bkg2 = ROOT.TH1F(f'{cname}_bkg2', f'{cname}_bkg2', *binning)
        for i, v in enumerate(bindata['bkg2']):
            bkg2.SetBinContent(i + 1, v)
        bkg2.Sumw2()

    if 'sig' in bindata:
        sig = ROOT.TH1F(f'{cname}_signal', f'{cname}_signal', *binning)
        for i, v in enumerate(bindata['sig']):
            sig.SetBinContent(i + 1, v)
        sig.Sumw2()
    f.Write()

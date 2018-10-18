import ROOT

import json
import sys

source_data = json.load(open(sys.argv[1]))
root_file = sys.argv[2]

binning = source_data['binning']
bindata = source_data['bindata']


f = ROOT.TFile(root_file, 'RECREATE')
data = ROOT.TH1F('data', 'data', *binning)
for i, v in enumerate(bindata['data']):
    data.SetBinContent(i + 1, v)
data.Sumw2()

bkg = ROOT.TH1F('bkg', 'bkg', *binning)
for i, v in enumerate(bindata['bkg']):
    bkg.SetBinContent(i + 1, v)
bkg.Sumw2()


if 'bkgerr' in bindata:
    bkgerr = ROOT.TH1F('bkgerr', 'bkgerr', *binning)

    # shapesys must be as multiplicative factor
    for i, v in enumerate(bindata['bkgerr']):
        bkgerr.SetBinContent(i + 1, v / bkg.GetBinContent(i + 1))
    bkgerr.Sumw2()

sig = ROOT.TH1F('sig', 'sig', *binning)
for i, v in enumerate(bindata['sig']):
    sig.SetBinContent(i + 1, v)
sig.Sumw2()
f.Write()

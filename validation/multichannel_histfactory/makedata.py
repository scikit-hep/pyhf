import ROOT

import json
import sys

source_data = json.load(open(sys.argv[1]))
root_file = sys.argv[2]

f = ROOT.TFile(root_file, 'RECREATE')

for cname, channel_def in source_data['channels'].iteritems():
    print('CH', cname)
    binning = channel_def['binning']
    bindata = channel_def['bindata']

    data = ROOT.TH1F('{}_data'.format(cname), '{}_data'.format(cname), *binning)
    for i, v in enumerate(bindata['data']):
        data.SetBinContent(i + 1, v)
    data.Sumw2()

    print(data.GetName())

    bkg = ROOT.TH1F('{}_bkg'.format(cname), '{}_bkg'.format(cname), *binning)
    for i, v in enumerate(bindata['bkg']):
        bkg.SetBinContent(i + 1, v)
    bkg.Sumw2()

    if 'bkgerr' in bindata:
        bkgerr = ROOT.TH1F(
            '{}_bkgerr'.format(cname), '{}_bkgerr'.format(cname), *binning
        )

        # shapesys must be as multiplicative factor
        for i, v in enumerate(bindata['bkgerr']):
            bkgerr.SetBinContent(i + 1, v / bkg.GetBinContent(i + 1))
        bkgerr.Sumw2()

    if 'sig' in bindata:
        sig = ROOT.TH1F('{}_signal'.format(cname), '{}_signal'.format(cname), *binning)
        for i, v in enumerate(bindata['sig']):
            sig.SetBinContent(i + 1, v)
        sig.Sumw2()
    f.Write()

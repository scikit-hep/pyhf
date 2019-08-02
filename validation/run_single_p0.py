import ROOT
import sys

infile = sys.argv[1]

infile = ROOT.TFile.Open(infile)
workspace = infile.Get("combined")
data = workspace.data("obsData")

sbModel = workspace.obj("ModelConfig")
poi = sbModel.GetParametersOfInterest().first()
poi.setVal(1)
sbModel.SetSnapshot(ROOT.RooArgSet(poi))

bModel = sbModel.Clone()
bModel.SetName("bonly")
poi.setVal(0)
bModel.SetSnapshot(ROOT.RooArgSet(poi))

ac = ROOT.RooStats.AsymptoticCalculator(data, sbModel, bModel)
ac.SetPrintLevel(10)
ac.SetOneSidedDiscovery(True)

result = ac.GetHypoTest()
pnull_obs = result.NullPValue()
palt_obs = result.AlternatePValue()
pnull_exp = []
for sigma in [-2, -1, 0, 1, 2]:
    usecls = 0
    pnull_exp.append(ac.GetExpectedPValues(pnull_obs, palt_obs, sigma, usecls))

import json

print(json.dumps({'p0_obs': pnull_obs, 'p0_exp': pnull_exp}, sort_keys=True, indent=4))

import ROOT
import sys

infile = sys.argv[1]

infile = ROOT.TFile.Open(infile)
workspace = infile.Get("combined")
data = workspace.data("obsData")

sbModel = workspace.obj("ModelConfig")
poi = sbModel.GetParametersOfInterest().first()

sbModel.SetSnapshot(ROOT.RooArgSet(poi))

bModel = sbModel.Clone()
bModel.SetName("bonly")
poi.setVal(0)
bModel.SetSnapshot(ROOT.RooArgSet(poi))

ac = ROOT.RooStats.AsymptoticCalculator(data, bModel, sbModel)
ac.SetPrintLevel(10)
ac.SetOneSided(True)
ac.SetQTilde(True)


calc = ROOT.RooStats.HypoTestInverter(ac)
calc.SetConfidenceLevel(0.95)
calc.UseCLs(True)
calc.RunFixedScan(1, 1, 1)

result = calc.GetInterval()

index = 0

w = result.GetExpectedPValueDist(index)
v = w.GetSamplingDistribution()

CLs_obs = result.CLs(index)
CLs_exp = list(v)[3:-3]

import json

print(json.dumps({'CLs_obs': CLs_obs, 'CLs_exp': CLs_exp}, sort_keys=True, indent=4))

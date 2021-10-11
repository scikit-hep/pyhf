import json
import sys

import ROOT

if __name__ == "__main__":
    infile = sys.argv[1]

    infile = ROOT.TFile.Open(infile)
    workspace = infile.Get("combined")
    data = workspace.data("obsData")

    sb_model = workspace.obj("ModelConfig")
    poi = sb_model.GetParametersOfInterest().first()

    sb_model.SetSnapshot(ROOT.RooArgSet(poi))

    bkg_model = sb_model.Clone()
    bkg_model.SetName("bonly")
    poi.setVal(0)
    bkg_model.SetSnapshot(ROOT.RooArgSet(poi))

    calc = ROOT.RooStats.AsymptoticCalculator(data, bkg_model, sb_model)
    calc.SetPrintLevel(10)
    calc.SetOneSided(True)
    calc.SetQTilde(True)

    test_inverter = ROOT.RooStats.HypoTestInverter(calc)
    test_inverter.SetConfidenceLevel(0.95)
    test_inverter.UseCLs(True)
    test_inverter.RunFixedScan(1, 1, 1)

    result = test_inverter.GetInterval()

    index = 0
    dist = result.GetExpectedPValueDist(index)
    CLs_obs = result.CLs(index)
    CLs_exp = list(dist.GetSamplingDistribution())[3:-3]

    print(
        json.dumps({"CLs_obs": CLs_obs, "CLs_exp": CLs_exp}, sort_keys=True, indent=4)
    )

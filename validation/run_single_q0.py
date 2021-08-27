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
    poi.setVal(1)
    sb_model.SetSnapshot(ROOT.RooArgSet(poi))

    bkg_model = sb_model.Clone()
    bkg_model.SetName("bonly")
    poi.setVal(0)
    bkg_model.SetSnapshot(ROOT.RooArgSet(poi))

    calc = ROOT.RooStats.AsymptoticCalculator(data, sb_model, bkg_model)
    calc.SetPrintLevel(10)
    calc.SetOneSidedDiscovery(True)

    result = calc.GetHypoTest()
    pnull_obs = result.NullPValue()
    palt_obs = result.AlternatePValue()
    usecls = 0
    pnull_exp = [
        calc.GetExpectedPValues(pnull_obs, palt_obs, sigma, usecls)
        for sigma in [-2, -1, 0, 1, 2]
    ]

    print(
        json.dumps({"p0_obs": pnull_obs, "p0_exp": pnull_exp}, sort_keys=True, indent=4)
    )

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

    asymptotic_calc = ROOT.RooStats.AsymptoticCalculator(data, bkg_model, sb_model)
    asymptotic_calc.SetPrintLevel(10)
    asymptotic_calc.SetOneSided(True)
    asymptotic_calc.SetQTilde(True)

    test_inverter = ROOT.RooStats.HypoTestInverter(asymptotic_calc)
    test_inverter.RunFixedScan(51, 0, 5)
    test_inverter.SetConfidenceLevel(0.95)
    test_inverter.UseCLs(True)

    result = test_inverter.GetInterval()

    plot = ROOT.RooStats.HypoTestInverterPlot("plot", "plot", result)
    canvas = ROOT.TCanvas()
    canvas.SetLogy(False)
    plot.Draw("OBS EXP CLb 2CL")
    canvas.Draw()
    canvas.SaveAs("scan.pdf")

    print(f"observed: {result.UpperLimit()}")

    for n_sigma in [-2, -1, 0, 1, 2]:
        print(f"expected {n_sigma}: {result.GetExpectedUpperLimit(n_sigma)}")

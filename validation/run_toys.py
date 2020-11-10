import ROOT
import sys
import json


def run_toys_ROOT(infile, ntoys):
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

    ac = ROOT.RooStats.FrequentistCalculator(data, bModel, sbModel)
    ac.SetToys(ntoys, ntoys)

    profll = ROOT.RooStats.ProfileLikelihoodTestStat(bModel.GetPdf())
    profll.SetOneSidedDiscovery(False)
    profll.SetOneSided(True)
    ac.GetTestStatSampler().SetTestStatistic(profll)

    calc = ROOT.RooStats.HypoTestInverter(ac)
    calc.SetConfidenceLevel(0.95)
    calc.UseCLs(True)

    npoints = 2

    calc.RunFixedScan(npoints, 1.0, 1.2)

    result = calc.GetInterval()

    plot = ROOT.RooStats.HypoTestInverterPlot("plot", "plot", result)

    data = []
    for i in range(npoints):
        d = {
            "test_b": list(result.GetAltTestStatDist(i).GetSamplingDistribution()),
            "test_s": list(result.GetNullTestStatDist(i).GetSamplingDistribution()),
            "pvals": list(result.GetExpectedPValueDist(i).GetSamplingDistribution()),
        }
        data.append(d)

    json.dump(data, open("scan.json", "w"))

    canvas = ROOT.TCanvas()
    canvas.SetLogy(False)
    plot.Draw("OBS EXP CLb 2CL")
    canvas.GetListOfPrimitives().At(0).GetYaxis().SetRangeUser(0, 0.2)
    canvas.Draw()
    extensions = ["pdf", "png"]
    for ext in extensions:
        canvas.SaveAs(f"poi_scan_ROOT.{ext}")


if __name__ == "__main__":
    run_toys_ROOT(
        infile=sys.argv[1], ntoys=int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    )
    # run_toys_pyhf()

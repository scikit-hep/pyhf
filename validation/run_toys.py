import json
import numpy as np
import matplotlib.pyplot as plt
import pyhf
import pyhf.contrib.viz.brazil as brazil
import ROOT
import sys


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


def run_toys_pyhf(ntoys=2_000, seed=0):
    np.random.seed(seed)
    # with open("validation/xmlimport_input_bkg.json") as ws_json:
    with open("debug/issue_workpace/issue_ws.json") as ws_json:
        workspace = pyhf.Workspace(json.load(ws_json))

    model = workspace.model()
    data = workspace.data(model)

    n_points = 4
    test_mus = np.linspace(1.0, 1.2, n_points)
    fit_results = [
        pyhf.infer.hypotest(
            mu, data, model, return_expected_set=True, calctype="toybased", ntoys=ntoys
        )
        for mu in test_mus
    ]

    fig, ax = plt.subplots()
    brazil.plot_results(test_mus, fit_results, ax=ax)
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$\mathrm{CL}_{s}$")
    _buffer = 0.02
    ax.set_xlim(1.0 - _buffer, 1.2 + _buffer)
    ax.set_ylim(0.0, 0.2)

    extensions = ["pdf", "png"]
    for ext in extensions:
        fig.savefig(f"poi_scan_pyhf.{ext}")

    ax.set_ylim(1e-3, 0.2)
    ax.semilogy()

    for ext in extensions:
        fig.savefig(f"poi_scan_logy_pyhf.{ext}")


if __name__ == "__main__":
    run_toys_ROOT(
        infile=sys.argv[1], ntoys=int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    )
    run_toys_pyhf(ntoys=2_000)

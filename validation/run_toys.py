import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import ROOT

import pyhf
import pyhf.contrib.viz.brazil as brazil


def run_toys_ROOT(infile, ntoys):
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

    calc = ROOT.RooStats.FrequentistCalculator(data, bkg_model, sb_model)
    calc.SetToys(ntoys, ntoys)

    profile_ll = ROOT.RooStats.ProfileLikelihoodTestStat(bkg_model.GetPdf())
    profile_ll.SetOneSidedDiscovery(False)
    profile_ll.SetOneSided(True)
    calc.GetTestStatSampler().SetTestStatistic(profile_ll)

    test_inverter = ROOT.RooStats.HypoTestInverter(calc)
    test_inverter.SetConfidenceLevel(0.95)
    test_inverter.UseCLs(True)

    n_points = 2

    test_inverter.RunFixedScan(n_points, 1.0, 1.2)

    result = test_inverter.GetInterval()

    plot = ROOT.RooStats.HypoTestInverterPlot("plot", "plot", result)

    data = [
        {
            "test_b": list(result.GetAltTestStatDist(idx).GetSamplingDistribution()),
            "test_s": list(result.GetNullTestStatDist(idx).GetSamplingDistribution()),
            "pvals": list(result.GetExpectedPValueDist(idx).GetSamplingDistribution()),
        }
        for idx in range(n_points)
    ]

    with open("scan.json", "w", encoding="utf-8") as write_file:
        json.dump(data, write_file)

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
    with open("debug/issue_workpace/issue_ws.json", encoding="utf-8") as ws_json:
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

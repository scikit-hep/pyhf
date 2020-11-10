import sys
import numpy as np
import json
import matplotlib.pyplot as plt
import ROOT
import pyhf


def StandardHypoTestDemo(
    infile,
    ntoys,
    workspaceName="combined",
    modelSBName="ModelConfig",
    dataName="obsData",
):

    file = ROOT.TFile.Open(infile)
    w = file.Get(workspaceName)
    sbModel = w.obj(modelSBName)
    data = w.data(dataName)

    bModel = sbModel.Clone()
    bModel.SetName(modelSBName + "B_only")
    var = bModel.GetParametersOfInterest().first()
    oldval = var.getVal()
    var.setVal(0)
    bModel.SetSnapshot(ROOT.RooArgSet(var))
    var.setVal(oldval)

    var = sbModel.GetParametersOfInterest().first()
    sbModel.SetSnapshot(ROOT.RooArgSet(var))

    profll = ROOT.RooStats.ProfileLikelihoodTestStat(bModel.GetPdf())
    profll.SetOneSidedDiscovery(False)
    profll.SetOneSided(True)
    hypoCalc = ROOT.RooStats.FrequentistCalculator(data, bModel, sbModel)
    # profll.SetPrintLevel(2)

    print('by hand', profll.Evaluate(data, ROOT.RooArgSet(var)))

    hypoCalc.SetToys(ntoys, ntoys)

    sampler = hypoCalc.GetTestStatSampler()
    sampler.SetTestStatistic(profll)

    htr = hypoCalc.GetHypoTest()
    htr.SetPValueIsRightTail(True)
    htr.SetBackgroundAsAlt(True)

    ds = htr.GetNullDistribution()
    db = htr.GetAltDistribution()

    r = ROOT.RooStats.HypoTestResult()
    r.SetPValueIsRightTail(True)
    r.SetNullDistribution(ds)
    r.SetAltDistribution(db)

    values = [
        1 / (r.SetTestStatisticData(v), r.CLs())[-1]
        for v in ds.GetSamplingDistribution()
    ]
    print(values)
    values = np.percentile(
        values, [2.27501319, 15.86552539, 50.0, 84.13447461, 97.72498681]
    )
    print(values)

    htr.Print()
    plot = ROOT.RooStats.HypoTestPlot(htr, 100)
    plot.SetLogYaxis(True)
    plot.Draw()
    ROOT.gPad.SaveAs("plot.png")


def pyhf_version(ntoys=5000, seed=0):
    np.random.seed(seed)
    with open("validation/xmlimport_input_bkg.json") as ws_json:
        workspace = pyhf.Workspace(json.load(ws_json))

    model = workspace.model()
    data = workspace.data(model)
    toy_calculator = pyhf.infer.utils.create_calculator(
        "toybased",
        data,
        model,
        ntoys=ntoys,
    )
    test_mu = 1.0
    sig_plus_bkg_dist, bkg_dist = toy_calculator.distributions(test_mu)
    q_tilde = toy_calculator.teststatistic(test_mu)

    bins = np.linspace(0, 8, 100)

    fig, ax = plt.subplots()
    # Compare to ROOT's choice of test stat being NLL instead of 2*NLL
    ax.hist(
        bkg_dist.samples / 2.0,
        alpha=0.2,
        bins=bins,
        density=True,
        label=r"$f(\tilde{q}|0)$ Background",
    )
    ax.hist(
        sig_plus_bkg_dist.samples / 2.0,
        alpha=0.2,
        bins=bins,
        density=True,
        label=r"$f(\tilde{q}|1)$ Signal",
    )
    ax.axvline(q_tilde / 2, color="black", label="Observed test statistic")
    ax.semilogy()

    ax.set_xlabel(r"$\tilde{q}$")
    ax.set_ylabel(r"$f(\tilde{q}|\mu')$")
    ax.legend(loc="best")

    fig.savefig("pyhf_version.png")


if __name__ == "__main__":
    StandardHypoTestDemo(
        infile=sys.argv[1], ntoys=int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    )
    # pyhf_version()

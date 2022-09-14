import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import ROOT

import pyhf


def standard_hypo_test_demo(
    infile,
    ntoys,
    workspace_name="combined",
    sb_model_name="ModelConfig",
    data_name="obsData",
):

    file = ROOT.TFile.Open(infile)
    workspace = file.Get(workspace_name)
    sb_model = workspace.obj(sb_model_name)
    data = workspace.data(data_name)

    bkg_model = sb_model.Clone()
    bkg_model.SetName(sb_model_name + "B_only")
    _var = bkg_model.GetParametersOfInterest().first()
    _oldval = _var.getVal()
    _var.setVal(0)
    bkg_model.SetSnapshot(ROOT.RooArgSet(_var))
    _var.setVal(_oldval)

    _var = sb_model.GetParametersOfInterest().first()
    sb_model.SetSnapshot(ROOT.RooArgSet(_var))

    profile_ll = ROOT.RooStats.ProfileLikelihoodTestStat(bkg_model.GetPdf())
    profile_ll.SetOneSidedDiscovery(False)
    profile_ll.SetOneSided(True)
    calc = ROOT.RooStats.FrequentistCalculator(data, bkg_model, sb_model)

    print(f'by hand: {profile_ll.Evaluate(data, ROOT.RooArgSet(_var))}')

    calc.SetToys(ntoys, ntoys)

    sampler = calc.GetTestStatSampler()
    sampler.SetTestStatistic(profile_ll)

    hypo_test_result = calc.GetHypoTest()
    hypo_test_result.SetPValueIsRightTail(True)
    hypo_test_result.SetBackgroundAsAlt(True)

    dist_signal = hypo_test_result.GetNullDistribution()
    dist_bkg = hypo_test_result.GetAltDistribution()

    result = ROOT.RooStats.HypoTestResult()
    result.SetPValueIsRightTail(True)
    result.SetNullDistribution(dist_signal)
    result.SetAltDistribution(dist_bkg)

    values = [
        1 / (result.SetTestStatisticData(v), result.CLs())[-1]
        for v in dist_signal.GetSamplingDistribution()
    ]
    print(values)
    values = np.percentile(
        values, [2.27501319, 15.86552539, 50.0, 84.13447461, 97.72498681]
    )
    print(values)

    hypo_test_result.Print()
    plot = ROOT.RooStats.HypoTestPlot(hypo_test_result, 100)
    plot.SetLogYaxis(True)
    plot.Draw()
    ROOT.gPad.SaveAs("plot.png")


def pyhf_version(ntoys=5000, seed=0):
    np.random.seed(seed)
    with open("validation/xmlimport_input_bkg.json", encoding="utf-8") as ws_json:
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
    standard_hypo_test_demo(
        infile=sys.argv[1], ntoys=int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    )
    pyhf_version()

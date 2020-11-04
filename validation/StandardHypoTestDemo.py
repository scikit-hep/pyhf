import ROOT

def StandardHypoTestDemo(
    infile,
    ntoys,
    workspaceName = "combined",
    modelSBName = "ModelConfig",
    dataName = "obsData",
):

    file = ROOT.TFile.Open(infile);
    w = file.Get(workspaceName);
    sbModel = w.obj(modelSBName);
    data = w.data(dataName);

    bModel = sbModel.Clone();
    bModel.SetName(modelSBName+"B_only");
    var = (bModel.GetParametersOfInterest().first());
    oldval = var.getVal();
    var.setVal(0);
    bModel.SetSnapshot( ROOT.RooArgSet(var)  );
    var.setVal(oldval);

    var = sbModel.GetParametersOfInterest().first();
    oldval = var.getVal();
    sbModel.SetSnapshot( ROOT.RooArgSet(var) );

    profll = ROOT.RooStats.ProfileLikelihoodTestStat(bModel.GetPdf());
    profll.SetOneSidedDiscovery(False);
    profll.SetOneSided(True);
    hypoCalc = ROOT.RooStats.FrequentistCalculator(data, bModel, sbModel);
    # profll.SetPrintLevel(2)


    print('by hand',profll.Evaluate(data,ROOT.RooArgSet(var) ))

    hypoCalc.SetToys(ntoys, ntoys)

    sampler = hypoCalc.GetTestStatSampler();
    sampler.SetTestStatistic(profll);

    paramPointNull = ROOT.RooArgSet(sbModel.GetParametersOfInterest())
    
    htr = hypoCalc.GetHypoTest();
    htr.SetPValueIsRightTail(True);
    htr.SetBackgroundAsAlt(True);
    htr.Print(); 
    plot = ROOT.RooStats.HypoTestPlot(htr,100);
    plot.SetLogYaxis(True);
    plot.Draw();
    ROOT.gPad.SaveAs("plot.png");

import sys
if __name__ == '__main__':
    StandardHypoTestDemo(sys.argv[1],int(sys.argv[2]) if len(sys.argv) > 2 else 2000)

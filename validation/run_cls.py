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
calc.RunFixedScan(51, 0, 5)
calc.SetConfidenceLevel(0.95)
calc.UseCLs(True)


result = calc.GetInterval()

plot = ROOT.RooStats.HypoTestInverterPlot("plot", "plot", result)
c = ROOT.TCanvas()
c.SetLogy(False)
plot.Draw("OBS EXP CLb 2CL")
c.Draw()
c.SaveAs('scan.pdf')


print('observed: {}'.format(result.UpperLimit()))

for i in [-2, -1, 0, 1, 2]:
    print('expected {}: {}'.format(i, result.GetExpectedUpperLimit(i)))

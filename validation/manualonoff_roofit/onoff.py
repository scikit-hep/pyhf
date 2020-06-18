import json
import ROOT

d = json.load(open('data/source.json'))
nobs = d['bindata']['data'][0]
b = d['bindata']['bkg'][0]
deltab = d['bindata']['bkgerr'][0]
s = d['bindata']['sig'][0]

# derived data
tau = b / deltab / deltab
mobs = round(tau * b)

print('tau: {}, m: {}'.format(tau, mobs))

w = ROOT.RooWorkspace("w", True)

# -----------------

w.factory("prod:nsig(mu[1,0,10],s[1])")
w.factory("sum:nexp_sr(nsig,b[1,40,300])")
w.factory("Poisson:on_model(nobs_sr[0,1000],nexp_sr)")

# -----------------

w.var('s').setVal(s)
w.var('b').setVal(b)

w.var('s').setConstant(True)
w.var('nobs_sr').setVal(nobs)


w.factory("prod:nexp_cr(tau[1],b)")
w.factory("Poisson:off_model(nobs_cr[0,1000],nexp_cr)")
w.var('nobs_cr').setVal(mobs)
w.var('nobs_cr').setConstant(True)
w.var('tau').setVal(tau)
w.var('tau').setConstant(True)

w.factory("PROD:onoff(on_model,off_model)")


data = ROOT.RooDataSet(
    'data', 'data', ROOT.RooArgSet(w.var('nobs_sr'), w.var('nobs_cr'))
)
data.add(ROOT.RooArgSet(w.var('nobs_sr'), w.var('nobs_cr')))

getattr(w, 'import')(data)

modelConfig = ROOT.RooStats.ModelConfig(w)
modelConfig.SetPdf(w.pdf('onoff'))
modelConfig.SetParametersOfInterest(ROOT.RooArgSet(w.var('mu')))
modelConfig.SetNuisanceParameters(ROOT.RooArgSet(w.var('b')))
modelConfig.SetObservables(ROOT.RooArgSet(w.var('nobs_sr'), w.var('nobs_cr')))
modelConfig.SetGlobalObservables(ROOT.RooArgSet())
modelConfig.SetName("ModelConfig")
getattr(w, 'import')(modelConfig)

w.Print()


##### model building complete


sbModel = w.obj('ModelConfig')
poi = sbModel.GetParametersOfInterest().first()

sbModel.SetSnapshot(ROOT.RooArgSet(poi))

bModel = sbModel.Clone()
bModel.SetName("bonly")
poi.setVal(0)
bModel.SetSnapshot(ROOT.RooArgSet(poi))


ac = ROOT.RooStats.AsymptoticCalculator(data, bModel, sbModel)
ac.SetOneSided(True)
# ac.SetQTilde(False)
ac.SetPrintLevel(10)
ROOT.RooStats.AsymptoticCalculator.SetPrintLevel(10)

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

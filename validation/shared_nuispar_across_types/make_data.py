import ROOT

sig = 'sig', [3, 1]
nom = 'nom', [12, 13]

histo_up = 'hup', [14, 15]
histo_dn = 'hdn', [10, 11]

data = 'data', [15, 16]

f = ROOT.TFile.Open('data.root', 'recreate')


for n, h in [sig, nom, histo_up, histo_dn, data]:
    rh = ROOT.TH1F(n, n, 2, -0.5, 1.5)
    for i, c in enumerate(h):
        rh.SetBinContent(1 + i, c)
    rh.Sumw2()
    rh.Write()

f.Close()

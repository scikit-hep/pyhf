import json
import sys

import ROOT

if __name__ == "__main__":
    source_data = json.load(open(sys.argv[1]))
    root_file = sys.argv[2]

    binning = source_data["binning"]
    bin_data = source_data["bin_data"]

    out_file = ROOT.TFile(root_file, "RECREATE")
    data = ROOT.TH1F("data", "data", *binning)
    for idx, value in enumerate(bin_data["data"]):
        data.SetBinContent(idx + 1, value)
    data.Sumw2()

    bkg = ROOT.TH1F("bkg", "bkg", *binning)
    for idx, value in enumerate(bin_data["bkg"]):
        bkg.SetBinContent(idx + 1, value)
    bkg.Sumw2()

    if "bkgerr" in bin_data:
        bkgerr = ROOT.TH1F("bkgerr", "bkgerr", *binning)

        # shapesys must be as multiplicative factor
        for idx, value in enumerate(bin_data["bkgerr"]):
            bkgerr.SetBinContent(idx + 1, value / bkg.GetBinContent(idx + 1))
        bkgerr.Sumw2()

    sig = ROOT.TH1F("sig", "sig", *binning)
    for idx, value in enumerate(bin_data["sig"]):
        sig.SetBinContent(idx + 1, value)
    sig.Sumw2()

    out_file.Write()

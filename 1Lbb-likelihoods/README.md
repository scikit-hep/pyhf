# JSON Likelihoods for 1Lbb Analysis

The JSON likelihoods are serialized in this folder. This is done by providing a background-only workspace containing the signal/control channels at `BkgOnly.json` as well as patch files for each mass point on the signal phase-space explored in the analysis.

All [jsonpatches](http://jsonpatch.com/) are contained in the file `patchset.json`. Each patch is identified in `patchset.json` by the metadata field `"name": "C1N2_Wh_hbb_[m1]_[m2]"` where `m1` is the mass of both the lightest chargino and the next-to-lightest neutralino (which are assumed to be nearly mass degenerate), and `m2` is the mass of the lightest neutralino.

## Producing signal workspaces

As an example, we use [python jsonpatch](https://python-json-patch.readthedocs.io/en/latest/) to make the full json likelihood workspace for the signal point `C1N2_Wh_hbb_700_400`:

```
jsonpatch BkgOnly.json <(pyhf patchset extract patchset.json --name "C1N2_Wh_hbb_700_400") > C1N2_Wh_hbb_700_400.json
```

## Computing signal workspaces

For example, with [pyhf](https://scikit-hep.org/pyhf/), you can do any of the following:

```
pyhf cls BkgOnly.json --patch <(pyhf patchset extract patchset.json --name "C1N2_Wh_hbb_700_400")

jsonpatch BkgOnly.json <(pyhf patchset extract patchset.json --name "C1N2_Wh_hbb_700_400") | pyhf cls

pyhf cls C1N2_Wh_hbb_700_400.json
```


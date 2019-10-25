import pytest
import requests
import tarfile
import json
import os
import pyhf
import numpy as np


@pytest.fixture(scope='module')
def sbottom_likelihoods_download():
    """Download the sbottom likelihoods tarball from HEPData"""
    sbottom_HEPData_URL = "https://www.hepdata.net/record/resource/997020?view=true"
    targz_filename = "sbottom_workspaces.tar.gz"
    response = requests.get(sbottom_HEPData_URL, stream=True)
    assert response.status_code == 200
    with open(targz_filename, 'wb') as file:
        file.write(response.content)
    # Open as a tarfile
    yield tarfile.open(targz_filename, "r:gz")
    os.remove(targz_filename)


@pytest.fixture(scope='module')
def regionA_bkgonly_json(sbottom_likelihoods_download):
    """Extract the background only model from sbottom Region A"""
    tarfile = sbottom_likelihoods_download
    bkgonly_json_data = (
        tarfile.extractfile(tarfile.getmember("RegionA/BkgOnly.json"))
        .read()
        .decode("utf8")
    )
    return json.loads(bkgonly_json_data)


@pytest.fixture(scope='module')
def regionA_signal_patch_json(sbottom_likelihoods_download):
    """Extract a signal model from sbottom Region A"""
    tarfile = sbottom_likelihoods_download
    signal_patch_json_data = (
        tarfile.extractfile(tarfile.getmember("RegionA/patch.sbottom_1300_205_60.json"))
        .read()
        .decode("utf8")
    )
    return json.loads(signal_patch_json_data)


def test_sbottom_regionA(regionA_bkgonly_json, regionA_signal_patch_json):
    bkg_only = regionA_bkgonly_json
    signal_patch = regionA_signal_patch_json
    workspace = pyhf.workspace.Workspace(bkg_only)
    model = workspace.model(
        measurement_name=None,
        patches=[signal_patch],
        modifier_settings={
            'normsys': {'interpcode': 'code4'},
            'histosys': {'interpcode': 'code4p'},
        },
    )
    result = pyhf.utils.hypotest(
        1.0, workspace.data(model), model, qtilde=True, return_expected_set=True
    )
    CLs_obs = result[0].tolist()[0]
    CLs_exp = result[-1].ravel().tolist()
    assert CLs_obs == pytest.approx(0.2444363575448201, rel=1e-7)
    assert np.all(
        np.isclose(
            np.array(CLs_exp),
            np.array(
                [
                    0.0902252193974136,
                    0.1937841171543251,
                    0.3843236961508878,
                    0.6557759457699649,
                    0.8910421945189615,
                ]
            ),
            rtol=1e-7,
        )
    )

import pytest
import requests
import hashlib
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
    with open(targz_filename, "wb") as file:
        file.write(response.content)
    assert (
        hashlib.sha256(open(targz_filename, "rb").read()).hexdigest()
        == "9089b0e5fabba335bea4c94545ccca8ddd21289feeab2f85e5bcc8bada37be70"
    )
    # Open as a tarfile
    yield tarfile.open(targz_filename, "r:gz")
    os.remove(targz_filename)


# Factory as fixture pattern
@pytest.fixture
def get_json_from_tarfile():
    def _get_json_from_tarfile(tarfile, json_name):
        json_file = (
            tarfile.extractfile(tarfile.getmember(json_name)).read().decode("utf8")
        )
        return json.loads(json_file)

    return _get_json_from_tarfile


def calculate_CLs(bkgonly_json, signal_patch_json):
    """
    Calculate the observed CLs and the expected CLs band from a background only
    and signal patch.

    Args:
        bkgonly_json: The JSON for the background only model
        signal_patch_json: The JSON Patch for the signal model

    Returns:
        CLs_obs: The observed CLs value
        CLs_exp: List of the expected CLs value band
    """
    workspace = pyhf.workspace.Workspace(bkgonly_json)
    model = workspace.model(
        measurement_name=None,
        patches=[signal_patch_json],
        modifier_settings={
            'normsys': {'interpcode': 'code4'},
            'histosys': {'interpcode': 'code4p'},
        },
    )
    result = pyhf.utils.hypotest(
        1.0, workspace.data(model), model, qtilde=True, return_expected_set=True
    )
    return result[0].tolist()[0], result[-1].ravel().tolist()


def test_sbottom_regionA_1300_205_60(
    sbottom_likelihoods_download, get_json_from_tarfile
):
    sbottom_regionA_bkgonly_json = get_json_from_tarfile(
        sbottom_likelihoods_download, "RegionA/BkgOnly.json"
    )
    sbottom_regionA_1300_205_60_patch_json = get_json_from_tarfile(
        sbottom_likelihoods_download, "RegionA/patch.sbottom_1300_205_60.json"
    )
    CLs_obs, CLs_exp = calculate_CLs(
        sbottom_regionA_bkgonly_json, sbottom_regionA_1300_205_60_patch_json
    )
    assert CLs_obs == pytest.approx(0.2444363575448201, rel=1e-5)
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
            rtol=1e-5,
        )
    )


def test_sbottom_regionA_1400_950_60(
    sbottom_likelihoods_download, get_json_from_tarfile
):
    sbottom_regionA_bkgonly_json = get_json_from_tarfile(
        sbottom_likelihoods_download, "RegionA/BkgOnly.json"
    )
    sbottom_regionA_1400_950_60_patch_json = get_json_from_tarfile(
        sbottom_likelihoods_download, "RegionA/patch.sbottom_1400_950_60.json"
    )
    CLs_obs, CLs_exp = calculate_CLs(
        sbottom_regionA_bkgonly_json, sbottom_regionA_1400_950_60_patch_json
    )
    assert CLs_obs == pytest.approx(0.0213732548585359, rel=1e-5)
    assert np.all(
        np.isclose(
            np.array(CLs_exp),
            np.array(
                [
                    0.0026446861433130,
                    0.0139766677574991,
                    0.0649728508540893,
                    0.2364443957087021,
                    0.5744835529584968,
                ]
            ),
            rtol=1e-5,
        )
    )


def test_sbottom_regionA_1500_850_60(
    sbottom_likelihoods_download, get_json_from_tarfile
):
    sbottom_regionA_bkgonly_json = get_json_from_tarfile(
        sbottom_likelihoods_download, "RegionA/BkgOnly.json"
    )
    sbottom_regionA_1500_850_60_patch_json = get_json_from_tarfile(
        sbottom_likelihoods_download, "RegionA/patch.sbottom_1500_850_60.json"
    )
    CLs_obs, CLs_exp = calculate_CLs(
        sbottom_regionA_bkgonly_json, sbottom_regionA_1500_850_60_patch_json
    )
    assert CLs_obs == pytest.approx(0.0453677409764101, rel=1e-5)
    assert np.all(
        np.isclose(
            np.array(CLs_exp),
            np.array(
                [
                    0.0059847029077065,
                    0.0261035161266011,
                    0.1009398575261459,
                    0.3101988586187604,
                    0.6553686728646031,
                ]
            ),
            rtol=1e-5,
        )
    )


def test_sbottom_regionB_1000_205_60(
    sbottom_likelihoods_download, get_json_from_tarfile
):
    sbottom_regionB_bkgonly_json = get_json_from_tarfile(
        sbottom_likelihoods_download, "RegionB/BkgOnly.json"
    )
    sbottom_regionB_1000_205_60_patch_json = get_json_from_tarfile(
        sbottom_likelihoods_download, "RegionB/patch.sbottom_1000_205_60.json"
    )
    CLs_obs, CLs_exp = calculate_CLs(
        sbottom_regionB_bkgonly_json, sbottom_regionB_1000_205_60_patch_json
    )
    assert CLs_obs == pytest.approx(0.999346961987008, rel=1e-5)
    assert np.all(
        np.isclose(
            np.array(CLs_exp),
            np.array(
                [
                    0.998057935502826,
                    0.998751430736146,
                    0.999346535249686,
                    0.999764360117854,
                    0.999954715109718,
                ]
            ),
            rtol=1e-5,
        )
    )


def test_sbottom_regionC_1000_205_60(
    sbottom_likelihoods_download, get_json_from_tarfile
):
    sbottom_regionC_bkgonly_json = get_json_from_tarfile(
        sbottom_likelihoods_download, "RegionC/BkgOnly.json"
    )
    sbottom_regionC_1000_205_60_patch_json = get_json_from_tarfile(
        sbottom_likelihoods_download, "RegionC/patch.sbottom_1000_205_60.json"
    )
    CLs_obs, CLs_exp = calculate_CLs(
        sbottom_regionC_bkgonly_json, sbottom_regionC_1000_205_60_patch_json
    )
    assert CLs_obs == pytest.approx(0.9424009499519606, rel=1e-5)
    # TODO: Lower tolerance to 1e-5 once Python 2.7 is dropped
    assert np.all(
        np.isclose(
            np.array(CLs_exp),
            np.array(
                [
                    0.8906470732412857,
                    0.9280262743211622,
                    0.9614288407343238,
                    0.9857553063165135,
                    0.9971958190844394,
                ]
            ),
            rtol=1e-4,
        )
    )

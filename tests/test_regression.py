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
    sbottom_HEPData_URL = "https://doi.org/10.17182/hepdata.89408.v1/r2"
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
    result = pyhf.infer.hypotest(
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
    assert CLs_obs == pytest.approx(0.24443627759085326, rel=1e-5)
    assert np.all(
        np.isclose(
            np.array(CLs_exp),
            np.array(
                [
                    0.09022509053507759,
                    0.1937839194960632,
                    0.38432344933992,
                    0.6557757334303531,
                    0.8910420971601081,
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
    assert CLs_obs == pytest.approx(0.021373283911064852, rel=1e-5)
    assert np.all(
        np.isclose(
            np.array(CLs_exp),
            np.array(
                [
                    0.002644707461012826,
                    0.013976754489151644,
                    0.06497313811425813,
                    0.23644505123524753,
                    0.5744843501873754,
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
    assert CLs_obs == pytest.approx(0.04536774062150508, rel=1e-5)
    assert np.all(
        np.isclose(
            np.array(CLs_exp),
            np.array(
                [
                    0.0059847029077065295,
                    0.026103516126601122,
                    0.10093985752614597,
                    0.3101988586187604,
                    0.6553686728646031,
                ]
            ),
            rtol=1e-5,
        )
    )


def test_sbottom_regionB_1400_550_60(
    sbottom_likelihoods_download, get_json_from_tarfile
):
    sbottom_regionB_bkgonly_json = get_json_from_tarfile(
        sbottom_likelihoods_download, "RegionB/BkgOnly.json"
    )
    sbottom_regionB_1400_550_60_patch_json = get_json_from_tarfile(
        sbottom_likelihoods_download, "RegionB/patch.sbottom_1400_550_60.json"
    )
    CLs_obs, CLs_exp = calculate_CLs(
        sbottom_regionB_bkgonly_json, sbottom_regionB_1400_550_60_patch_json
    )
    assert CLs_obs == pytest.approx(0.9744675266677597, rel=1e-5)
    assert np.all(
        np.isclose(
            np.array(CLs_exp),
            np.array(
                [
                    0.9338879894557114,
                    0.9569045303300702,
                    0.9771296335437559,
                    0.9916370124133669,
                    0.9983701133999316,
                ]
            ),
            rtol=1e-5,
        )
    )


def test_sbottom_regionC_1600_850_60(
    sbottom_likelihoods_download, get_json_from_tarfile
):
    sbottom_regionC_bkgonly_json = get_json_from_tarfile(
        sbottom_likelihoods_download, "RegionC/BkgOnly.json"
    )
    sbottom_regionC_1600_850_60_patch_json = get_json_from_tarfile(
        sbottom_likelihoods_download, "RegionC/patch.sbottom_1600_850_60.json"
    )
    CLs_obs, CLs_exp = calculate_CLs(
        sbottom_regionC_bkgonly_json, sbottom_regionC_1600_850_60_patch_json
    )
    assert CLs_obs == pytest.approx(0.711023707425625, rel=1e-5)
    assert np.all(
        np.isclose(
            np.array(CLs_exp),
            np.array(
                [
                    0.2955492909588046,
                    0.4446885457298284,
                    0.6371473864200973,
                    0.8336149623750603,
                    0.9585901381554178,
                ]
            ),
            rtol=1e-5,
        )
    )

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


def extract_json_from_tarfile(tarfile, json_name):
    json_file = tarfile.extractfile(tarfile.getmember(json_name)).read().decode("utf8")
    return json.loads(json_file)


@pytest.fixture(scope='module')
def regionA_bkgonly_json(sbottom_likelihoods_download):
    """Extract the background only model from sbottom Region A"""
    return extract_json_from_tarfile(
        sbottom_likelihoods_download, "RegionA/BkgOnly.json"
    )


@pytest.fixture(scope='module')
def regionB_bkgonly_json(sbottom_likelihoods_download):
    """Extract the background only model from sbottom Region B"""
    return extract_json_from_tarfile(
        sbottom_likelihoods_download, "RegionB/BkgOnly.json"
    )


@pytest.fixture(scope='module')
def regionC_bkgonly_json(sbottom_likelihoods_download):
    """Extract the background only model from sbottom Region C"""
    return extract_json_from_tarfile(
        sbottom_likelihoods_download, "RegionC/BkgOnly.json"
    )


@pytest.fixture()
def regionA_signal_patch_json(sbottom_likelihoods_download):
    """Extract a signal model from sbottom Region A"""
    return extract_json_from_tarfile(
        sbottom_likelihoods_download, "RegionA/patch.sbottom_1300_205_60.json"
    )


@pytest.fixture()
def regionB_signal_patch_json(sbottom_likelihoods_download):
    """Extract a signal model from sbottom Region B"""
    return extract_json_from_tarfile(
        sbottom_likelihoods_download, "RegionB/patch.sbottom_1000_205_60.json"
    )


@pytest.fixture()
def regionC_signal_patch_json(sbottom_likelihoods_download):
    """Extract a signal model from sbottom Region C"""
    return extract_json_from_tarfile(
        sbottom_likelihoods_download, "RegionC/patch.sbottom_1000_205_60.json"
    )


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


def test_sbottom_regionA(regionA_bkgonly_json, regionA_signal_patch_json):
    CLs_obs, CLs_exp = calculate_CLs(regionA_bkgonly_json, regionA_signal_patch_json)
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


def test_sbottom_regionB(regionB_bkgonly_json, regionB_signal_patch_json):
    CLs_obs, CLs_exp = calculate_CLs(regionB_bkgonly_json, regionB_signal_patch_json)
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


def test_sbottom_regionC(regionC_bkgonly_json, regionC_signal_patch_json):
    CLs_obs, CLs_exp = calculate_CLs(regionC_bkgonly_json, regionC_signal_patch_json)
    assert CLs_obs == pytest.approx(0.9424021663134358, rel=1e-5)
    # TODO: Lower tolerance to 1e-5 once Python 2.7 is dropped
    assert np.all(
        np.isclose(
            np.array(CLs_exp),
            np.array(
                [
                    0.8906506884676416,
                    0.9280287127442725,
                    0.9614301796189283,
                    0.9857558128338463,
                    0.9971959212073871,
                ]
            ),
            rtol=1e-4,
        )
    )

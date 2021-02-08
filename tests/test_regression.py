import pytest
import pyhf
import numpy as np


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
        1.0, workspace.data(model), model, test_stat="qtilde", return_expected_set=True
    )
    return result[0].tolist(), result[-1]


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
    assert CLs_obs == pytest.approx(0.24444183656462842, rel=1e-5)
    assert np.all(
        np.isclose(
            np.array(CLs_exp),
            np.array(
                [
                    0.09023416633313781,
                    0.19379784038559986,
                    0.3843408317367939,
                    0.655790687844951,
                    0.8910489537320908,
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
    assert CLs_obs == pytest.approx(0.021373129575592117, rel=1e-5)

    assert np.all(
        np.isclose(
            np.array(CLs_exp),
            np.array(
                [
                    0.0026445915209657935,
                    0.013976282781787152,
                    0.06497157578576944,
                    0.23644148599230647,
                    0.5744800142422956,
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
    assert CLs_obs == pytest.approx(0.04536628480130834, rel=1e-5)

    assert np.all(
        np.isclose(
            np.array(CLs_exp),
            np.array(
                [
                    0.00598364810953375,
                    0.026100020461594278,
                    0.10093042667790517,
                    0.31018132834142287,
                    0.655351306669138,
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
    assert CLs_obs == pytest.approx(0.9944320696159622, rel=1e-5)

    assert np.all(
        np.isclose(
            np.array(CLs_exp),
            np.array(
                [
                    0.9939882032779436,
                    0.9961316425804301,
                    0.9979736724671936,
                    0.9992686756878328,
                    0.9998593342137401,
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
    assert CLs_obs == pytest.approx(0.711026385080708, rel=1e-5)
    assert np.all(
        np.isclose(
            np.array(CLs_exp),
            np.array(
                [
                    0.29555621162757173,
                    0.44469567377123215,
                    0.6371533630019872,
                    0.8336184150255452,
                    0.9585912011479805,
                ]
            ),
            rtol=1e-5,
        )
    )

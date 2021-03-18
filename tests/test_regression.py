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
            rtol=2e-5,
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
                    0.0026445531093281147,
                    0.013976126501170727,
                    0.06497105816950004,
                    0.23644030478043676,
                    0.5744785776763938,
                ],
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
    assert CLs_obs == pytest.approx(0.045367205665400624, rel=1e-5)
    assert np.all(
        np.isclose(
            np.array(CLs_exp),
            np.array(
                [
                    0.00598431785676406,
                    0.026102240062850574,
                    0.10093641492218848,
                    0.31019245951964736,
                    0.6553623337518385,
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

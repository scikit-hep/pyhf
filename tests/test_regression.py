import pyhf
import requests
import tarfile
import json
import os
import pytest


# def test_sbottom_download():
#     # curl -sL hepdata.net/record/resourc…
#     # | tar -O -xzv RegionA/BkgOnly.json
#     # | pyhf cls --patch <(curl -sL hepdata.net/record/resourc…
#     # | tar -O -xzv RegionA/patch.sbottom_1300_205_60.json)
#     sbottom_HEPData_URL = "https://www.hepdata.net/record/resource/997020?view=true"
#     response = requests.get(sbottom_HEPData_URL, stream=True)
#     response.raise_for_status()
#
#     with open('/home/mcf/Code/GitHub/pyhf/output.gz', 'wb') as handle:
#         for block in response.iter_content(1024):
#             handle.write(block)
#     # response.content is a GzipFile object
#     # if downloaded with webbrowser is named HEPData_workspaces.tar.gz
#     print(tarfile.is_tarfile(response.content))
#     # tar_file = tarfile.open(name="archive", mode="r:gz", fileobj=response.content)
#     # print(tar_file)


@pytest.fixture(scope='module')
def sbottom_likelihoods_download():
    sbottom_HEPData_URL = "https://www.hepdata.net/record/resource/997020?view=true"
    targz_filename = "sbottom_workspaces.tar.gz"
    # Download the tar.gz of the likelihoods
    # file_path = os.path.join(tmp_path.strpath, sbottom_HEPData_URL)
    # file_name = tmp_path.join(targz_filename)
    # response = requests.get(tmp_path.join(sbottom_HEPData_URL), stream=True)
    response = requests.get(sbottom_HEPData_URL, stream=True)
    assert response.status_code == 200
    with open(targz_filename, 'wb') as file:
        file.write(response.content)
    # Open as a tarfile
    yield tarfile.open(targz_filename, "r:gz")
    os.remove(targz_filename)


@pytest.fixture(scope='module')
def regionA_bkgonly_json(sbottom_likelihoods_download):
    tarfile = sbottom_likelihoods_download
    bkgonly_json_data = (
        tarfile.extractfile(tarfile.getmember("RegionA/BkgOnly.json"))
        .read()
        .decode("utf8")
    )
    return json.loads(bkgonly_json_data)


@pytest.fixture(scope='module')
def regionA_signal_patch_json(sbottom_likelihoods_download):
    tarfile = sbottom_likelihoods_download
    signal_patch_json_data = (
        tarfile.extractfile(tarfile.getmember("RegionA/patch.sbottom_1300_205_60.json"))
        .read()
        .decode("utf8")
    )
    return json.loads(signal_patch_json_data)


def test_gzip_file(regionA_bkgonly_json, regionA_signal_patch_json):
    bkg_json = regionA_bkgonly_json
    signal_json = regionA_signal_patch_json
    workspace = pyhf.workspace.Workspace(bkg_json)
    patched = workspace.model(
        measurement_name=None,
        patches=[signal_json],
        modifier_settings={
            'normsys': {'interpcode': 'code4'},
            'histosys': {'interpcode': 'code4p'},
        },
    )
    result = pyhf.utils.hypotest(
        1.0, workspace.data(patched), patched, qtilde=True, return_expected_set=True
    )
    result = {'CLs_obs': result[0].tolist()[0], 'CLs_exp': result[-1].ravel().tolist()}
    print(json.dumps(result, indent=4, sort_keys=True))

    # command line
    # {
    #     "CLs_exp": [
    #         0.09022521939741368,
    #         0.19378411715432514,
    #         0.3843236961508878,
    #         0.6557759457699649,
    #         0.8910421945189615
    #     ],
    #     "CLs_obs": 0.24443635754482018
    # }
    # pytest
    # {'CLs_obs': 0.24443635754482018, 'CLs_exp': [0.09022521939741368, 0.19378411715432514, 0.3843236961508878, 0.6557759457699649, 0.8910421945189615]}

    # print(s)
    # TODO: Patch bkg_json with signal_json to create workspace
    # Then validate
    # assert pyhf.utils.validate(workspace, 'workspace.json')

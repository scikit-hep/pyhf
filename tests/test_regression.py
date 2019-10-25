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
    s = json.dumps(bkg_json, indent=4, sort_keys=True)
    s = json.dumps(signal_json, indent=4, sort_keys=True)
    # print(s)
    # TODO: Patch bkg_json with signal_json to create workspace
    # Then validate
    # assert pyhf.utils.validate(workspace, 'workspace.json')

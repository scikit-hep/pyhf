import tarfile
import zipfile

import pytest

from pyhf.contrib.utils import download
from pyhf.exceptions import InvalidArchive, InvalidArchiveHost


def test_download_untrusted_archive_host(tmpdir, requests_mock):
    archive_url = "https://www.pyhfthisdoesnotexist.org"
    requests_mock.get(archive_url)

    with pytest.raises(InvalidArchiveHost):
        download(archive_url, tmpdir.join("likelihoods").strpath)


@pytest.mark.parametrize(
    "archive_url",
    [
        "https://www.hepdata.net/record/resource/1408476?view=true",
    ],
)
def test_download_invalid_archive(tmpdir, requests_mock, archive_url):
    requests_mock.get(archive_url, status_code=404)

    with pytest.raises(InvalidArchive):
        download(archive_url, tmpdir.join("likelihoods").strpath)


# @pytest.mark.parametrize(
#     'compress', [True, False]
# )
@pytest.mark.parametrize(
    "archive_url",
    [
        "https://www.hepdata.net/record/resource/1408476?view=true",
    ],
)
def test_download_archive_type(tmpdir, mocker, requests_mock, archive_url):
    # Give BytesIO a tarfile
    with tarfile.open(tmpdir.join("test_tar.tar.gz").strpath, mode="w:gz") as archive:
        with open(tmpdir.join("test_tar_file.txt").strpath, "wb") as write_file:
            write_file.write(b"tarfile test")
        archive.add(tmpdir.join("test_tar_file.txt").strpath)

    requests_mock.get(
        archive_url,
        content=open(tmpdir.join("test_tar.tar.gz").strpath, "rb").read(),
    )
    download(archive_url, tmpdir.join("likelihoods").strpath)

    # Give BytesIO a zipfile
    with zipfile.ZipFile(tmpdir.join("test_zip.zip").strpath, "w") as archive:
        with open(tmpdir.join("test_zip_file.txt").strpath, "w") as write_file:
            write_file.write("zipfile test")
        archive.write(tmpdir.join("test_zip_file.txt").strpath)

    requests_mock.get(
        archive_url,
        content=open(tmpdir.join("test_zip.zip").strpath, "rb").read(),
    )
    download(archive_url, tmpdir.join("likelihoods").strpath)

    # TODO: This is hacky and ugly. Try to clean up with something else
    requests_mock.get(archive_url)
    mocker.patch(
        "io.BytesIO",
        return_value=open(tmpdir.join("test_tar.tar.gz").strpath, "rb"),
    )
    with pytest.raises(InvalidArchive):
        download(archive_url, tmpdir.join("likelihoods").strpath)

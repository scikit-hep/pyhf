import tarfile
import zipfile
from pathlib import Path
from shutil import rmtree

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


def test_download_compress(tmpdir, requests_mock):
    archive_url = "https://www.hepdata.net/record/resource/1408476?view=true"
    requests_mock.get(archive_url)

    download(archive_url, tmpdir.join("likelihoods").strpath, compress=True)


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
    # Run without and with existing output_directory to cover both
    # cases of the shutil.rmtree logic
    rmtree(Path(tmpdir.join("likelihoods").strpath))
    download(archive_url, tmpdir.join("likelihoods").strpath)  # without
    download(archive_url, tmpdir.join("likelihoods").strpath)  # with

    # Give BytesIO a zipfile (using same requests_mock as previous) but have
    # zipfile.is_zipfile reject it
    mocker.patch("zipfile.is_zipfile", return_value=False)
    with pytest.raises(InvalidArchive):
        download(archive_url, tmpdir.join("likelihoods").strpath)

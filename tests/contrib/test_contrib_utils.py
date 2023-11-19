import tarfile
import zipfile
from pathlib import Path
from shutil import rmtree

import pytest

from pyhf.contrib.utils import download
from pyhf.exceptions import InvalidArchive, InvalidArchiveHost


@pytest.fixture(scope="function")
def tarfile_path(tmp_path):
    with open(
        tmp_path.join("test_file.txt").strpath, "w", encoding="utf-8"
    ) as write_file:
        write_file.write("test file")
    with tarfile.open(
        tmp_path.join("test_tar.tar.gz").strpath, mode="w:gz", encoding="utf-8"
    ) as archive:
        archive.add(tmp_path.join("test_file.txt").strpath)
    return Path(tmp_path.join("test_tar.tar.gz").strpath)


@pytest.fixture(scope="function")
def tarfile_uncompressed_path(tmp_path):
    with open(
        tmp_path.join("test_file.txt").strpath, "w", encoding="utf-8"
    ) as write_file:
        write_file.write("test file")
    with tarfile.open(
        tmp_path.join("test_tar.tar").strpath, mode="w", encoding="utf-8"
    ) as archive:
        archive.add(tmp_path.join("test_file.txt").strpath)
    return Path(tmp_path.join("test_tar.tar").strpath)


@pytest.fixture(scope="function")
def zipfile_path(tmp_path):
    with open(
        tmp_path.join("test_file.txt").strpath, "w", encoding="utf-8"
    ) as write_file:
        write_file.write("test file")
    with zipfile.ZipFile(tmp_path.join("test_zip.zip").strpath, "w") as archive:
        archive.write(tmp_path.join("test_file.txt").strpath)
    return Path(tmp_path.join("test_zip.zip").strpath)


def test_download_untrusted_archive_host(tmp_path, requests_mock):
    archive_url = "https://www.pyhfthisdoesnotexist.org"
    requests_mock.get(archive_url)

    with pytest.raises(InvalidArchiveHost):
        download(archive_url, tmp_path.join("likelihoods").strpath)


def test_download_invalid_archive(tmp_path, requests_mock):
    archive_url = "https://www.hepdata.net/record/resource/1408476?view=true"
    requests_mock.get(archive_url, status_code=404)

    with pytest.raises(InvalidArchive):
        download(archive_url, tmp_path.join("likelihoods").strpath)


def test_download_compress(tmp_path, requests_mock):
    archive_url = "https://www.hepdata.net/record/resource/1408476?view=true"
    requests_mock.get(archive_url)

    download(archive_url, tmp_path.join("likelihoods").strpath, compress=True)


def test_download_archive_type(
    tmp_path,
    mocker,
    requests_mock,
    tarfile_path,
    tarfile_uncompressed_path,
    zipfile_path,
):
    archive_url = "https://www.hepdata.net/record/resource/1408476?view=true"
    output_directory = tmp_path.join("likelihoods").strpath
    # Give BytesIO a tarfile
    requests_mock.get(archive_url, content=open(tarfile_path, "rb").read())
    download(archive_url, output_directory)

    # Give BytesIO an uncompressed tarfile
    requests_mock.get(archive_url, content=open(tarfile_uncompressed_path, "rb").read())
    download(archive_url, output_directory)

    # Give BytesIO a zipfile
    requests_mock.get(archive_url, content=open(zipfile_path, "rb").read())
    # Run without and with existing output_directory to cover both
    # cases of the shutil.rmtree logic
    rmtree(Path(output_directory))
    download(archive_url, output_directory)  # without
    download(archive_url, output_directory)  # with

    # Give BytesIO a zipfile (using same requests_mock as previous) but have
    # zipfile.is_zipfile reject it
    mocker.patch("zipfile.is_zipfile", return_value=False)
    with pytest.raises(InvalidArchive):
        download(archive_url, output_directory)


def test_download_archive_force(tmp_path, requests_mock, tarfile_path):
    archive_url = "https://www.cern.ch/record/resource/123456789"
    requests_mock.get(
        archive_url, content=open(tarfile_path, "rb").read(), status_code=200
    )

    with pytest.raises(InvalidArchiveHost):
        download(archive_url, tmp_path.join("likelihoods").strpath, force=False)

    download(archive_url, tmp_path.join("likelihoods").strpath, force=True)

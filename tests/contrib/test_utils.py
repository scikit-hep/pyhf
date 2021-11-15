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

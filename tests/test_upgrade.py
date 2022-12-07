import pyhf
import pyhf.schema
import json
import logging
import pytest


def test_upgrade_bad_version(datadir):
    with pytest.raises(ValueError):
        pyhf.schema.upgrade(to_version='0.9.0')


def test_upgrade_to_latest(datadir):
    ws = json.load(open(datadir.joinpath("workspace_1.0.0.json"), encoding="utf-8"))
    pyhf.schema.upgrade().workspace(ws)

    ps = json.load(open(datadir.joinpath("workspace_1.0.0.json"), encoding="utf-8"))
    pyhf.schema.upgrade().patchset(ps)


def test_1_0_0_workspace(datadir, caplog, monkeypatch):
    """
    Test upgrading a workspace from 1.0.0
    """
    spec = json.load(open(datadir.joinpath("workspace_1.0.0.json"), encoding="utf-8"))

    monkeypatch.setitem(pyhf.schema.versions, 'workspace.json', '1.0.1')
    with caplog.at_level(logging.INFO, 'pyhf.schema'):
        pyhf.schema.validate(spec, 'workspace.json', version='1.0.0')
        assert 'Specification requested version 1.0.0' in caplog.text

    caplog.clear()

    new_spec = pyhf.schema.upgrade(to_version='1.0.1').workspace(spec)
    assert new_spec['version'] == '1.0.1'


def test_1_0_0_patchset(datadir, caplog, monkeypatch):
    """
    Test upgrading a patchset from 1.0.0
    """
    spec = json.load(open(datadir.joinpath("patchset_1.0.0.json"), encoding="utf-8"))

    monkeypatch.setitem(pyhf.schema.versions, 'patchset.json', '1.0.1')
    with caplog.at_level(logging.INFO, 'pyhf.schema'):
        pyhf.schema.validate(spec, 'patchset.json', version='1.0.0')
        assert 'Specification requested version 1.0.0' in caplog.text

    caplog.clear()

    new_spec = pyhf.schema.upgrade(to_version='1.0.1').patchset(spec)
    assert new_spec['version'] == '1.0.1'

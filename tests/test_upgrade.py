import pyhf
import pyhf.schema
import json
import logging


def test_1_0_0_workspace(datadir, caplog):
    """
    Test upgrading a workspace from 1.0.0
    """
    spec = json.load(open(datadir.joinpath("workspace_1.0.0.json")))

    with caplog.at_level(logging.INFO, 'pyhf.schema'):
        pyhf.schema.validate(spec, 'workspace.json', version='1.0.0')
        assert 'Specification requested version 1.0.0' in caplog.text

    caplog.clear()

    new_spec = pyhf.schema.upgrade_workspace(spec)
    assert new_spec['version'] == '1.0.1'
    with caplog.at_level(logging.INFO, 'pyhf.schema'):
        pyhf.schema.validate(new_spec, 'workspace.json', version='1.0.1')
        assert caplog.text == ''


def test_1_0_0_patchset(datadir, caplog):
    """
    Test upgrading a patchset from 1.0.0
    """
    spec = json.load(open(datadir.joinpath("patchset_1.0.0.json")))

    with caplog.at_level(logging.INFO, 'pyhf.schema'):
        pyhf.schema.validate(spec, 'patchset.json', version='1.0.0')
        assert 'Specification requested version 1.0.0' in caplog.text

    caplog.clear()

    new_spec = pyhf.schema.upgrade_patchset(spec)
    assert new_spec['version'] == '1.0.1'
    with caplog.at_level(logging.INFO, 'pyhf.schema'):
        pyhf.schema.validate(new_spec, 'patchset.json', version='1.0.1')
        assert caplog.text == ''

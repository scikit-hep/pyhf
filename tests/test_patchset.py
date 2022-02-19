import pyhf
import pytest
import pyhf.exceptions
import pyhf.patchset
import json
from unittest import mock


@pytest.fixture(
    scope='function',
    params=['patchset_good.json', 'patchset_good_2_patches.json'],
    ids=['patchset_good.json', 'patchset_good_2_patches.json'],
)
def patchset(datadir, request):
    with open(datadir.joinpath(request.param), encoding="utf-8") as spec_file:
        spec = json.load(spec_file)
    return pyhf.PatchSet(spec)


@pytest.fixture(scope='function')
def patch():
    return pyhf.patchset.Patch(
        {'metadata': {'name': 'test', 'values': [1.0, 2.0, 3.0]}, 'patch': {}}
    )


@pytest.mark.parametrize(
    'patchset_file',
    [
        'patchset_bad_empty_patches.json',
        'patchset_bad_no_version.json',
        'patchset_bad_wrong_valuetype.json',
    ],
)
def test_patchset_invalid_spec(datadir, patchset_file):
    with open(datadir.joinpath(patchset_file), encoding="utf-8") as patchset_spec_file:
        patchsetspec = json.load(patchset_spec_file)
    with pytest.raises(pyhf.exceptions.InvalidSpecification):
        pyhf.PatchSet(patchsetspec)


@pytest.mark.parametrize(
    'patchset_file',
    [
        'patchset_bad_duplicate_patch_name.json',
        'patchset_bad_duplicate_patch_values.json',
        'patchset_bad_wrong_values_multiplicity.json',
    ],
)
def test_patchset_bad(datadir, patchset_file):
    with open(datadir.joinpath(patchset_file), encoding="utf-8") as patchset_spec_file:
        patchsetspec = json.load(patchset_spec_file)
    with pytest.raises(pyhf.exceptions.InvalidPatchSet):
        pyhf.PatchSet(patchsetspec)


def test_patchset_attributes(patchset):
    assert 'hepdata' in patchset.references
    assert patchset.description == "signal patchset for the SUSY Multi-b-jet analysis"
    assert len(patchset.digests) == 1
    assert patchset.digests['md5'] == "098f6bcd4621d373cade4e832627b4f6"
    assert patchset.labels == ["mass_stop", "mass_neutralino"]
    assert patchset.version == "1.0.0"


def test_patchset_get_patch_by_name(patchset):
    assert patchset['Gtt_2100_5000_800']


def test_patchset_get_patch_by_values(patchset):
    assert patchset[2100, 800]
    assert patchset[(2100, 800)]
    assert patchset[[2100, 800]]


def test_patchset_get_nonexisting_patch(patchset):
    with pytest.raises(pyhf.exceptions.InvalidPatchLookup) as excinfo:
        patchset.__getitem__('nonexisting_patch')
    assert 'No patch associated with' in str(excinfo.value)
    assert 'nonexisting_patch' in str(excinfo.value)


def test_patchset_iterable(patchset):
    assert iter(patchset)
    assert list(iter(patchset))
    assert len(list(iter(patchset))) >= 1


def test_patchset_len(patchset):
    assert len(patchset) == len(list(iter(patchset)))
    assert len(patchset) == len(patchset.patches)


def test_patchset_repr(patchset):
    assert repr(patchset)
    if len(patchset) == 1:
        assert 'PatchSet object with 1 patch at' in repr(patchset)
    else:
        assert f'PatchSet object with {len(patchset)} patches at' in repr(patchset)


def test_patchset_verify(datadir):
    with open(
        datadir.joinpath("example_patchset.json"), encoding="utf-8"
    ) as patchset_file:
        patchset = pyhf.PatchSet(json.load(patchset_file))
    with open(datadir.joinpath("example_bkgonly.json"), encoding="utf-8") as bkg_file:
        ws = pyhf.Workspace(json.load(bkg_file))
    assert patchset.verify(ws) is None


def test_patchset_verify_failure(datadir):
    with open(
        datadir.joinpath("example_patchset.json"), encoding="utf-8"
    ) as patchset_file:
        patchset = pyhf.PatchSet(json.load(patchset_file))
    with pytest.raises(pyhf.exceptions.PatchSetVerificationError):
        assert patchset.verify({})


def test_patchset_apply(datadir):
    with open(
        datadir.joinpath("example_patchset.json"), encoding="utf-8"
    ) as patchset_file:
        patchset = pyhf.PatchSet(json.load(patchset_file))
    with open(datadir.joinpath("example_bkgonly.json"), encoding="utf-8") as bkg_file:
        ws = pyhf.Workspace(json.load(bkg_file))
    with mock.patch('pyhf.patchset.PatchSet.verify') as m:
        assert m.call_count == 0
        assert patchset.apply(ws, 'patch_channel1_signal_syst1')
        assert m.call_count == 1


def test_patch_hashable(patch):
    assert patch.name == 'test'
    assert isinstance(patch.values, tuple)
    assert patch.values == (1.0, 2.0, 3.0)


def test_patch_repr(patch):
    assert repr(patch)
    assert "Patch object 'test(1.0, 2.0, 3.0)' at" in repr(patch)


def test_patch_equality(patch):
    assert patch == patch
    assert patch != object()


def test_patchset_get_string_values(datadir):
    with open(
        datadir.joinpath("patchset_good_stringvalues.json"), encoding="utf-8"
    ) as patchset_file:
        patchset = pyhf.PatchSet(json.load(patchset_file))

    assert patchset["Gtt_2100_5000_800"]
    assert patchset["Gbb_2200_5000_800"]
    assert patchset[[2100, 800, "Gtt"]]
    assert patchset[[2100, 800, "Gbb"]]

import pyhf
import pytest
import pyhf.exceptions
import pyhf.patchset
import json


@pytest.fixture(
    scope='function',
    params=['patchset_good.json', 'patchset_good_2_patches.json'],
    ids=['patchset_good.json', 'patchset_good_2_patches.json'],
)
def patchset(datadir, request):
    spec = json.load(open(datadir.join(request.param)))
    return pyhf.Patchset(spec)


@pytest.fixture(scope='function')
def patch():
    return pyhf.patchset.Patch(
        {'metadata': {'name': 'test', 'values': [1.0, 2.0, 3.0]}, 'patch': {}}
    )


def test_patchset_invalid_spec(datadir):
    patchsetspec = json.load(open(datadir.join('patchset_bad_empty_patches.json')))
    with pytest.raises(pyhf.exceptions.InvalidSpecification):
        pyhf.Patchset(patchsetspec)


@pytest.mark.parametrize(
    'patchset_file',
    [
        'patchset_bad_duplicate_patch_name.json',
        'patchset_bad_duplicate_patch_values.json',
    ],
)
def test_patchset_bad(datadir, patchset_file):
    patchsetspec = json.load(open(datadir.join(patchset_file)))
    with pytest.raises(pyhf.exceptions.InvalidPatchset):
        pyhf.Patchset(patchsetspec)


def test_patchset_attributes(patchset):
    assert patchset.analysis_id == "SUSY-2018-23"
    assert patchset.description == "signal patchset for the SUSY Multi-b-jet analysis"
    assert len(patchset.digests) == 1
    assert patchset.digests['md5'] == "098f6bcd4621d373cade4e832627b4f6"
    assert patchset.labels == ["mass_stop", "mass_neutralino"]


def test_patchset_get_patch_by_name(patchset):
    assert patchset['Gtt_2100_5000_800']


def test_patchset_get_patch_by_values(patchset):
    assert patchset[2100, 800]
    assert patchset[(2100, 800)]
    assert patchset[[2100, 800]]


def test_patchset_get_nonexisting_patch(patchset):
    with pytest.raises(pyhf.exceptions.InvalidPatchLookup) as excinfo:
        patch = patchset['nonexisting_patch']
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
        assert 'Patchset object with 1 patch at' in repr(patchset)
    else:
        assert f'Patchset object with {len(patchset)} patches at' in repr(patchset)


def test_patch_bashable(patch):
    assert patch.name == 'test'
    assert isinstance(patch.values, tuple)
    assert patch.values == (1.0, 2.0, 3.0)


def test_patch_repr(patch):
    assert repr(patch)
    assert "Patch object 'test(1.0, 2.0, 3.0)' at" in repr(patch)

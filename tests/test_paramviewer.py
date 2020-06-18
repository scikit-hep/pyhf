import pyhf
from pyhf.parameters import ParamViewer


def test_paramviewer_simple_nonbatched(backend):
    pars = pyhf.tensorlib.astensor([1, 2, 3, 4, 5, 6, 7])

    parshape = pyhf.tensorlib.shape(pars)

    view = ParamViewer(
        parshape,
        {'hello': {'slice': slice(0, 2)}, 'world': {'slice': slice(5, 7)}},
        ['world', 'hello'],
    )
    par_slice = view.get(pars)
    assert pyhf.tensorlib.tolist(par_slice[slice(2, 4)]) == [1, 2]

    assert pyhf.tensorlib.tolist(par_slice[slice(0, 2)]) == [6, 7]

    assert pyhf.tensorlib.tolist(par_slice) == [6, 7, 1, 2]


def test_paramviewer_order(sbottom_likelihoods_download, get_json_from_tarfile):
    lhood = get_json_from_tarfile(sbottom_likelihoods_download, "RegionA/BkgOnly.json")
    patch = get_json_from_tarfile(
        sbottom_likelihoods_download, "RegionA/patch.sbottom_1300_205_60.json"
    )
    workspace = pyhf.workspace.Workspace(lhood)
    model = workspace.model(patches=[patch])

    pv = ParamViewer((model.config.npars,), model.config.par_map, [])
    assert list(pv.allpar_viewer.names) == model.config.par_order


def test_paramviewer_simple_batched(backend):
    pars = pyhf.tensorlib.astensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

    parshape = pyhf.tensorlib.shape(pars)

    view = ParamViewer(
        parshape,
        {'hello': {'slice': slice(0, 2)}, 'world': {'slice': slice(3, 4)}},
        ['world', 'hello'],
    )
    par_slice = view.get(pars)

    assert isinstance(view.index_selection, list)
    assert all(
        [len(x) == 3 for x in view.index_selection]
    )  # first dimension is batch dim

    assert pyhf.tensorlib.shape(par_slice) == (3, 3)
    assert pyhf.tensorlib.tolist(par_slice[slice(1, 3)]) == [[1, 5, 9], [2, 6, 10]]
    assert pyhf.tensorlib.tolist(par_slice[slice(0, 1)]) == [[4, 8, 12]]

    assert pyhf.tensorlib.tolist(par_slice) == [[4, 8, 12], [1, 5, 9], [2, 6, 10]]

import pyhf
from pyhf.parameters import ParamViewer


def test_paramviewer_simple_nonbatched(backend):
    pars = pyhf.tensorlib.astensor([1, 2, 3, 4, 5, 6, 7])

    parshape = pyhf.tensorlib.shape(pars)

    view = ParamViewer(
        parshape,
        {'hello': {'slice': slice(0, 2)}, 'world': {'slice': slice(5, 7)}},
        ['hello', 'world'],
    )
    par_slice = view.get(pars)
    assert pyhf.tensorlib.tolist(par_slice[view.slices[0]]) == [1, 2]

    assert pyhf.tensorlib.tolist(par_slice[view.slices[1]]) == [6, 7]

    assert pyhf.tensorlib.tolist(par_slice) == [1, 2, 6, 7]


def test_paramviewer_simple_batched(backend):
    pars = pyhf.tensorlib.astensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

    parshape = pyhf.tensorlib.shape(pars)

    view = ParamViewer(
        parshape,
        {'hello': {'slice': slice(0, 2)}, 'world': {'slice': slice(3, 4)}},
        ['hello', 'world'],
    )
    par_slice = view.get(pars)

    assert isinstance(view.index_selection, list)
    assert all(
        [len(x) == 3 for x in view.index_selection]
    )  # first dimension is batch dim

    assert pyhf.tensorlib.shape(par_slice) == (3, 3)
    assert pyhf.tensorlib.tolist(par_slice[view.slices[0]]) == [[1, 5, 9], [2, 6, 10]]
    assert pyhf.tensorlib.tolist(par_slice[view.slices[1]]) == [[4, 8, 12]]

    assert pyhf.tensorlib.tolist(par_slice) == [[1, 5, 9], [2, 6, 10], [4, 8, 12]]

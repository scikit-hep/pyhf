import pyhf
import numpy as np
from pyhf.paramview import ParamViewer


def test_paramviewer_simple_nonbatched(backend):
    pars = pyhf.tensorlib.astensor([1, 2, 3, 4, 5, 6, 7])

    parshape = pyhf.tensorlib.shape(pars)

    v = ParamViewer(
        parshape,
        {'hello': {'slice': slice(0, 2)}, 'world': {'slice': slice(5, 7)}},
        ['hello', 'world'],
    )
    sl = v.get(pars)
    assert pyhf.tensorlib.tolist(sl[v.slices[0]]) == [1, 2]

    assert pyhf.tensorlib.tolist(sl[v.slices[1]]) == [6, 7]

    assert pyhf.tensorlib.tolist(sl) == [1, 2, 6, 7]


def test_paramviewer_simple_batched(backend):
    pars = pyhf.tensorlib.astensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

    parshape = pyhf.tensorlib.shape(pars)

    v = ParamViewer(
        parshape,
        {'hello': {'slice': slice(0, 2)}, 'world': {'slice': slice(3, 4)}},
        ['hello', 'world'],
    )
    sl = v.get(pars)

    assert v.index_selection == 0

    assert pyhf.tensorlib.shape(sl) == (3, 3)
    assert pyhf.tensorlib.tolist(sl[v.slices[0]]) == [[1, 5, 9], [2, 6, 10]]
    assert pyhf.tensorlib.tolist(sl[v.slices[1]]) == [[4, 8, 12]]

    assert pyhf.tensorlib.tolist(sl) == [[1, 5, 9], [2, 6, 10], [4, 8, 12]]

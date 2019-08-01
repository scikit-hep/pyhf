import pyhf
import numpy as np
from pyhf.paramview import ParamViewer


def test_paramviewer(backend):
    pars = pyhf.tensorlib.astensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

    nbatch = pyhf.tensorlib.shape(pars)[0]
    npars = pyhf.tensorlib.shape(pars)[1]

    v = ParamViewer((nbatch, npars), {'hello': {'slice': slice(0, 2)}}, ['hello'])
    sl = v.get(pars)
    assert pyhf.tensorlib.tolist(sl[v.slices[0]]) == [[1, 5, 9], [2, 6, 10]]

    # v = ParamViewer((nbatch, npars), {'hello': {'slice': slice(0, 2)}}, 'hello')
    # sl = v.get(pars)
    # assert pyhf.tensorlib.tolist(sl) == [[1, 5, 9], [2, 6, 10]]

    # pars = [1, 2, 3, 4]
    # npars = len(pars)
    # v = ParamViewer((npars,), {'hello': {'slice': slice(0, 2)}}, ['hello'])
    # sl = v.get(pars)
    # assert pyhf.tensorlib.tolist(sl[0]) == [1]

    # v = ParamViewer((npars,), {'hello': {'slice': slice(0, 2)}}, 'hello')
    # sl = v.get(pars)
    # assert pyhf.tensorlib.tolist(sl) == [1, 2]

    # pars = [
    #     list(range(5))
    # ]*10
    # print('pars',pars)
    # v = ParamViewer(
    #     (10,5),
    #     {'first': {'slice': slice(0, 2)},
    #     'second': {'slice': slice(2,5)}}, ['first','second'],
    #     regular = False
    # )
    # sl = v.get_slice(pars)
    # assert np.all(np.isclose(pyhf.tensorlib.tolist(sl[0]),[[0,1]]*10))
    # assert np.all(np.isclose(pyhf.tensorlib.tolist(sl[1]),[[2,3,4]]*10))

    # pars = [
    #     list(range(4))
    # ]*10
    # print('pars',pars)
    # v = ParamViewer(
    #     (10,5),
    #     {'first': {'slice': slice(0, 2)},
    #     'second': {'slice': slice(0, 4)}}, ['first','second'],
    # )

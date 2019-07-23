import pyhf
from pyhf.paramview import ParamViewer

def test_paramviewer(backend):
    pars = [
        [1,2,3,4],
        [5,6,7,8],
        [9,10,11,12],
    ]

    nbatch = len(pars)
    npars  = len(pars[0])


    v = ParamViewer((nbatch,npars), {'hello': {'slice': slice(0,2)}}, ['hello'])
    sl = v.get_slice(pars)
    assert pyhf.tensorlib.tolist(sl[0]) == [
        [1,2],
        [5,6],
        [9,10],
    ]


    v = ParamViewer((nbatch,npars), {'hello': {'slice': slice(0,2)}}, 'hello')
    sl = v.get_slice(pars)
    assert pyhf.tensorlib.tolist(sl) == [
        [1,2],
        [5,6],
        [9,10],
    ]



    pars = [1,2,3,4]
    npars  = len(pars)
    v = ParamViewer((npars,), {'hello': {'slice': slice(0,2)}}, ['hello'])
    sl = v.get_slice(pars)
    assert pyhf.tensorlib.tolist(sl[0]) == [1,2]

    v = ParamViewer((npars,), {'hello': {'slice': slice(0,2)}}, 'hello')
    sl = v.get_slice(pars)
    assert pyhf.tensorlib.tolist(sl) == [1,2]

from pyhf.tensor.pytorch_backend import pytorch_backend
from pyhf.tensor.numpy_backend import numpy_backend

def test_common_tensor_backends():
    for tb in [numpy_backend(), pytorch_backend()]:
        assert tb.tolist(tb.astensor([1,2,3])) == [1,2,3]
        assert tb.tolist(tb.ones((2,3))) == [[1,1,1],[1,1,1]]
        assert tb.tolist(tb.sum([[1,2,3],[4,5,6]], axis = 0)) == [5,7,9]
        assert tb.tolist(tb.product([[1,2,3],[4,5,6]], axis = 0)) == [4,10,18]
        assert tb.tolist(tb.power([1,2,3],[1,2,3])) == [1,4,27]
        assert tb.tolist(tb.divide([4,9,16],[2,3,4])) == [2,3,4]
        assert tb.tolist(tb.outer([1,2,3],[4,5,6])) == [[4,5,6],[8,10,12],[12,15,18]]
        assert tb.tolist(tb.sqrt([4,9,16])) == [2,3,4]

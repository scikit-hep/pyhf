from pyhf.tensor.pytorch_backend import pytorch_backend
from pyhf.tensor.numpy_backend import numpy_backend
from pyhf.tensor.tensorflow_backend import tensorflow_backend
import tensorflow as tf

def test_common_tensor_backends():
    tf_sess = tf.Session()
    for tb in [numpy_backend(), pytorch_backend(), tensorflow_backend(session = tf_sess)]:
        assert tb.tolist(tb.astensor([1,2,3])) == [1,2,3]
        assert tb.tolist(tb.ones((2,3))) == [[1,1,1],[1,1,1]]
        assert tb.tolist(tb.sum([[1,2,3],[4,5,6]], axis = 0)) == [5,7,9]
        assert tb.tolist(tb.product([[1,2,3],[4,5,6]], axis = 0)) == [4,10,18]
        assert tb.tolist(tb.power([1,2,3],[1,2,3])) == [1,4,27]
        assert tb.tolist(tb.divide([4,9,16],[2,3,4])) == [2,3,4]
        assert tb.tolist(tb.outer([1,2,3],[4,5,6])) == [[4,5,6],[8,10,12],[12,15,18]]
        assert tb.tolist(tb.sqrt([4,9,16])) == [2,3,4]
        assert tb.tolist(tb.stack([tb.astensor([1,2,3]),tb.astensor([4,5,6])])) == [[1,2,3],[4,5,6]]
        assert tb.tolist(tb.concatenate([tb.astensor([1,2,3]),tb.astensor([4,5,6])])) == [1,2,3,4,5,6]
        assert tb.tolist(tb.log(tb.exp([2,3,4]))) == [2,3,4]
        assert tb.tolist(tb.where(
            tb.astensor([1,0,1]),
            tb.astensor([1,1,1]),
            tb.astensor([2,2,2]))) == [1,2,1]

        assert list(map(tb.tolist,tb.simple_broadcast(
            tb.astensor([1,1,1]),
            tb.astensor([2]),
            tb.astensor([3,3,3])))) == [[1,1,1],[2,2,2],[3,3,3]]

from pyhf.tensor.common import _TensorViewer


def test_tensorviewer(backend):
    tb, _ = backend
    tv = _TensorViewer(
        [tb.astensor([0, 4, 5]), tb.astensor([1, 2, 3]), tb.astensor([6]),],
        names=['zzz', 'aaa', 'x'],
    )

    data = tb.astensor(tb.astensor(list(range(7))) * 10, dtype='int')

    a = [tb.tolist(x) for x in tv.split(data, selection=['aaa'])]
    assert a == [[10, 20, 30]]

    a = [tb.tolist(x) for x in tv.split(data, selection=['aaa', 'zzz'])]
    assert a == [[10, 20, 30], [0, 40, 50]]

    a = [tb.tolist(x) for x in tv.split(data, selection=['zzz', 'aaa'])]
    assert a == [[0, 40, 50], [10, 20, 30]]

    a = [tb.tolist(x) for x in tv.split(data, selection=['x', 'aaa'])]
    assert a == [[60], [10, 20, 30]]

    a = [tb.tolist(x) for x in tv.split(data, selection=[])]
    assert a == []

    a = [tb.tolist(x) for x in tv.split(data)]
    assert a == [[0, 40, 50], [10, 20, 30], [60]]

    subviewer = _TensorViewer(
        [tb.astensor([0]), tb.astensor([1, 2, 3]),], names=['x', 'aaa']
    )
    assert tb.tolist(subviewer.stitch(tv.split(data, ['x', 'aaa']))) == [60, 10, 20, 30]

    subviewer = _TensorViewer(
        [tb.astensor([0, 1, 2]), tb.astensor([3]),], names=['aaa', 'x']
    )
    assert tb.tolist(subviewer.stitch(tv.split(data, ['aaa', 'x']))) == [10, 20, 30, 60]

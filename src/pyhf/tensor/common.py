from .. import default_backend, get_backend
from .. import events


class _TensorViewer(object):
    def __init__(self, indices, batch_size=None, names=None):
        # self.partition_indices has the "target" indices
        # of the stitched vector. In order to  .gather()
        # an concatennation of source arrays into the
        # desired form, one needs to gather on the "sorted"
        # indices
        # >>> source = np.asarray([9,8,7,6])
        # >>> target = np.asarray([2,1,3,0])
        # >>> source[target.argsort()]
        # array([6, 8, 9, 7])

        self.batch_size = batch_size
        self.names = names
        self._partition_indices = indices
        _concat_indices = default_backend.astensor(
            default_backend.concatenate(self._partition_indices), dtype='int'
        )
        self._sorted_indices = default_backend.tolist(_concat_indices.argsort())

        self._precompute()
        events.subscribe('tensorlib_changed')(self._precompute)

    def _precompute(self):
        tensorlib, _ = get_backend()
        self.sorted_indices = tensorlib.astensor(self._sorted_indices, dtype='int')
        self.partition_indices = [
            tensorlib.astensor(idx, dtype='int') for idx in self._partition_indices
        ]
        if self.names:
            self.name_map = dict(zip(self.names, self.partition_indices))

    def stitch(self, data):
        tensorlib, _ = get_backend()
        assert len(self.partition_indices) == len(data)

        data = tensorlib.concatenate(data, axis=-1)
        if len(tensorlib.shape(data)) == 1:
            stitched = tensorlib.gather(data, self.sorted_indices)
        else:
            data = tensorlib.einsum('...j->j...', data)
            stitched = tensorlib.gather(data, self.sorted_indices)
            stitched = tensorlib.einsum('j...->...j', stitched)
        return stitched

    def split(self, data, selection=None):
        tensorlib, _ = get_backend()
        indices = (
            self.partition_indices
            if selection is None
            else [self.name_map[n] for n in selection]
        )
        if len(tensorlib.shape(data)) == 1:
            return [tensorlib.gather(data, idx) for idx in indices]
        data = tensorlib.einsum('...j->j...', tensorlib.astensor(data))
        return [
            tensorlib.einsum('j...->...j', tensorlib.gather(data, idx))
            for idx in indices
        ]


def _tensorviewer_from_slices(slices, names, batch_size):
    target_slices = []
    start = 0
    for sl in slices:
        stop = start + (sl.stop - sl.start)
        target_slices.append(slice(start, stop))
        start = stop

    db = default_backend
    ranges = []
    for sl in target_slices:
        ranges.append(db.astensor(range(sl.start, sl.stop)))
    if not ranges:
        return None
    return _TensorViewer(ranges, names=names, batch_size=batch_size)

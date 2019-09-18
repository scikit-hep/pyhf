from .. import default_backend, get_backend
from .. import events


class TensorViewer(object):
    def __init__(self, indices):
        # self.partition_indices has the "target" indices
        # of the stitched vector. In order to  .gather()
        # an concatennation of source arrays into the
        # desired form, one needs to gather on the "sorted"
        # indices
        # >>> source = np.asarray([9,8,7,6])
        # >>> target = np.asarray([2,1,3,0])
        # >>> source[target.argsort()]
        # array([6, 8, 9, 7])

        self.partition_indices = indices
        a = default_backend.astensor(
            default_backend.concatenate(self.partition_indices), dtype='int'
        )
        self._sorted_indices = default_backend.tolist(a.argsort())

        self._precompute()
        events.subscribe('tensorlib_changed')(self._precompute)

    def _precompute(self):
        tensorlib, _ = get_backend()
        self.sorted_indices = tensorlib.astensor(self._sorted_indices)

    def stitch(self, data):
        tensorlib, _ = get_backend()
        assert len(self.partition_indices) == len(data)

        data = tensorlib.concatenate(data, axis=-1)
        # move event axis up front
        data = tensorlib.einsum('...j->j...', data)
        stitched = tensorlib.gather(
            data, tensorlib.astensor(self.sorted_indices, dtype='int')
        )
        return tensorlib.einsum('j...->...j', stitched)

    def split(self, data):
        tensorlib, _ = get_backend()
        data = tensorlib.astensor(data)
        data = tensorlib.einsum('...j->j...', data)
        split = [
            tensorlib.einsum('j...->...j', tensorlib.gather(data, idx))
            for idx in self.partition_indices
        ]
        return split

from .. import get_backend, default_backend, events
from ..tensor.common import _TensorViewer


def index_helper(name, tensor_shape, batch_shape, par_map):
    if isinstance(name, list):
        return [index_helper(x, tensor_shape, batch_shape, par_map) for x in name]
    x = list(range(int(default_backend.product(tensor_shape))))
    indices = default_backend.reshape(x, tensor_shape)
    parfield_slice = tuple(slice(None, None) for x in batch_shape) + (
        par_map[name]['slice'],
    )
    indices = indices[parfield_slice]
    return default_backend.tolist(indices)


class ParamViewer(object):
    """
    Helper class to extract parameter data from possibly batched input
    """

    def __init__(self, shape, par_map, selection):
        db = default_backend
        self.shape = shape
        self.selection = selection

        # prepares names and per-parset ranges
        # in the order or the parameters
        names, indices, starts = list(
            zip(
                *sorted(
                    [
                        (
                            k,
                            db.astensor(range(v['slice'].start, v['slice'].stop)),
                            v['slice'].start,
                        )
                        for k, v in par_map.items()
                    ],
                    key=lambda x: x[2],
                )
            )
        )

        self.batch = shape[0] if len(shape) > 1 else None

        # a tensor viewer that can split and stitch parameters
        self.allpar_viewer = _TensorViewer(indices, names=names, batch_size=self.batch)

        # to combine the selected
        # parameters into a overall tensor
        # we need to prep some ranges
        slices = []
        ranges = []
        start = 0
        for s in selection:
            sl = par_map[s]['slice']
            stop = start + (sl.stop - sl.start)
            ranges.append(db.astensor(range(start, stop)))
            slices.append(slice(start, stop))
            start = stop

        # used in tests
        self.slices = slices

        if self.selection:
            # a tensor viewer that can split and stitch the selected parameters
            self.selected_viewer = _TensorViewer(ranges, batch_size=self.batch)

        self._precompute()
        events.subscribe('tensorlib_changed')(self._precompute)

    def _precompute(self):
        tensorlib, _ = get_backend()

        theshape = tensorlib.product(tensorlib.astensor(self.shape, dtype='int'))
        flat_indices = tensorlib.astensor(list(range(int(theshape))), dtype='int')
        all_indices = tensorlib.reshape(flat_indices, self.shape)
        self.selected = self.allpar_viewer.split(all_indices, selection=self.selection)

        # LH: just self.selected but as python lists
        self.index_selection = [
            tensorlib.tolist(tensorlib.astensor(x, dtype='int')) for x in self.selected
        ]

        if self.selection:
            stitched = self.selected_viewer.stitch(self.selected)

            # LH: the transpose is here so that modifier code doesn't have to do it
            self.indices_concatenated = tensorlib.astensor(
                tensorlib.einsum('ij->ji', stitched) if self.batch else stitched,
                dtype='int',
            )

    def get(self, data, indices=None):
        if not self.index_selection:
            return None
        tensorlib, _ = get_backend()
        indices = indices if indices is not None else self.indices_concatenated
        return tensorlib.gather(tensorlib.reshape(data, (-1,)), indices)

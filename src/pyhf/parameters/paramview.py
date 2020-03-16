from .. import get_backend, default_backend, events
from ..tensor.common import _TensorViewer


def _tensorviewer_from_parmap(par_map, batch_size):
    db = default_backend
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
    return _TensorViewer(indices, names=names, batch_size=batch_size)


def _tensorviewer_from_slices(slices, batch_size):
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
        return (target_slices, None)
    return target_slices, _TensorViewer(ranges, batch_size=batch_size)


class ParamViewer(object):
    """
    Helper class to extract parameter data from possibly batched input
    """

    def __init__(self, shape, par_map, selection):
        db = default_backend
        self.shape = shape
        self.selection = selection

        self.batch = shape[0] if len(shape) > 1 else None

        # a tensor viewer that can split and stitch parameters
        self.allpar_viewer = _tensorviewer_from_parmap(par_map, self.batch)

        # a tensor viewer that can split and stitch the selected parameters
        self.slices, self.selected_viewer = _tensorviewer_from_slices(
            [par_map[s]['slice'] for s in selection] , self.batch
        )

        self._precompute()
        events.subscribe('tensorlib_changed')(self._precompute)

    def _precompute(self):
        tensorlib, _ = get_backend()

        fullsize = tensorlib.product(tensorlib.astensor(self.shape, dtype='int'))
        flat_indices = tensorlib.astensor(list(range(int(fullsize))), dtype='int')
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

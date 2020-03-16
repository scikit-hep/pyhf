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


def extract_index_access(
    baseviewer, subviewer, indices,
):
    tensorlib, _ = get_backend()

    index_selection = []
    stitched = None
    indices_concatenated = None
    if subviewer:
        index_selection = baseviewer.split(indices, selection=subviewer.names)
        stitched = subviewer.stitch(index_selection)

        # the transpose is here so that modifier code doesn't have to do it
        indices_concatenated = tensorlib.astensor(
            tensorlib.einsum('ij->ji', stitched)
            if len(tensorlib.shape(stitched)) > 1
            else stitched,
            dtype='int',
        )
    return index_selection, stitched, indices_concatenated


class ParamViewer(object):
    """
    Helper class to extract parameter data from possibly batched input
    """

    def __init__(self, shape, par_map, selection):

        batch = shape[0] if len(shape) > 1 else None

        fullsize = default_backend.product(default_backend.astensor(shape))
        flat_indices = default_backend.astensor(range(int(fullsize)), dtype='int')
        self._all_indices = default_backend.reshape(flat_indices, shape)

        # a tensor viewer that can split and stitch parameters
        self.allpar_viewer = _tensorviewer_from_parmap(par_map, batch)

        # a tensor viewer that can split and stitch the selected parameters
        self.selected_viewer = _tensorviewer_from_slices(
            [par_map[s]['slice'] for s in selection], selection, batch
        )

        self._precompute()
        events.subscribe('tensorlib_changed')(self._precompute)

    def _precompute(self):
        tensorlib, _ = get_backend()

        self.all_indices = tensorlib.astensor(self._all_indices)
        (
            self.index_selection,
            self.stitched,
            self.indices_concatenated,
        ) = extract_index_access(
            self.allpar_viewer, self.selected_viewer, self.all_indices
        )

    def get(self, data, indices=None):
        if not self.index_selection:
            return None
        tensorlib, _ = get_backend()
        indices = indices if indices is not None else self.indices_concatenated
        return tensorlib.gather(tensorlib.reshape(data, (-1,)), indices)

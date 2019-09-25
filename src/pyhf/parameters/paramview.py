from .. import get_backend, default_backend, events


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

    def __init__(self, tensor_shape, par_map, name):
        self.tensor_shape = tensor_shape
        self.batch_shape = tensor_shape[:-1]
        # for more general batch shapes we need to revisit
        assert len(self.batch_shape) <= 1
        self._par_map = par_map
        self._index_selection = index_helper(
            name, tensor_shape, self.batch_shape, self._par_map
        )

        if self._index_selection:
            cat = default_backend.astensor(
                default_backend.concatenate(self._index_selection, axis=-1), dtype='int'
            )
            # index_selection is
            #   batched:   list of (batch_dim, slice size) tensors
            #   unbatched: list of (slice size,) tensors
            # concatenated is
            #   batched: (batch_dim, sum of slice sizes)
            #   unbatched: (sum of slice sizes, )
            # indices_concatenated  is
            #   batched: (sum of slice size, batch dim)
            #   unbatched: (sum of slice size, )
            if self.batch_shape:
                self._indices_concatenated = default_backend.einsum('ij->ji', cat)
            else:
                self._indices_concatenated = cat

        else:
            self._indices_concatenated = None

        last = 0
        sl = []
        for s in [
            self._par_map[x]['slice'].stop - self._par_map[x]['slice'].start
            for x in (name if isinstance(name, list) else [name])
        ]:
            sl.append(slice(last, last + s))
            last += s
        self._slices = sl

        self._precompute()
        events.subscribe('tensorlib_changed')(self._precompute)

    def _precompute(self):
        tensorlib, _ = get_backend()
        if self._indices_concatenated is not None:
            self.indices_concatenated = tensorlib.astensor(
                self._indices_concatenated, dtype='int'
            )

    def __repr__(self):
        return '({} with [{}] batched: {})'.format(
            self.tensor_shape,
            ' '.join(list(self._par_map.keys())),
            bool(self.batch_shape),
        )

    @property
    def index_selection(self):
        """
        Returns:
            indices into parameter field accordig to requested subset of parameters
            list of (batch_size, parset_size) tensors
        """
        return self._index_selection

    @property
    def slices(self):
        """
        Returns:
            list index slices to retrieve a subset of the requested parameters
        """
        return self._slices

    def get(self, tensor, indices=None):
        """
        Args:
            tensor (`tensor`): the data tensor to extract a view from
            indices (`tensor`): an optional index selection (default behavior is to use self.indices_concatenated)
        Returns:
            filtered set of parameter field/array :
                type when batched: (sum of slice sizes, batchsize) tensor
                type when not batched: (sum of slice sizes, ) tensors
        """
        tensorlib, _ = get_backend()
        if not self.batch_shape:
            return tensorlib.gather(
                tensor, indices if indices is not None else self.indices_concatenated
            )
        return tensorlib.gather(
            tensorlib.reshape(tensor, (-1,)),
            indices if indices is not None else self.indices_concatenated,
        )

from . import get_backend, default_backend, events


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
        self.par_map = par_map
        self.index_selection = index_helper(
            name, tensor_shape, self.batch_shape, self.par_map
        )

        last = 0
        sl = []
        for s in [par_map[x]['slice'].stop-par_map[x]['slice'].start for x in (name if isinstance(name,list) else [name])]:
            sl.append(slice(last,last+s))
            last +=s
        self.slices = sl\


        self._precompute()
        events.subscribe('tensorlib_changed')(self._precompute)


    
    def _precompute(self):
        tensorlib, _ = get_backend()
        if self.index_selection:
            cat = tensorlib.concatenate([tensorlib.astensor(x, dtype = 'int') for x in self.index_selection],axis=1)
            self.indices_concatenated = tensorlib.einsum('ij->ji',  cat)
    
    def __repr__(self):
        return '({} with [{}] batched: {})'.format(
            self.tensor_shape,
            ' '.join(list(self.par_map.keys())),
            bool(self.batch_shape),
        )

    def get(self, tensor):
        """
        Returns:
            list of parameter slices:
                type when batched: list of (batchsize, slicesize,) tensors
                type when not batched: list of (slicesize, ) tensors
        """
        tensorlib, _ = get_backend()
        result = tensorlib.gather(tensorlib.reshape(tensor, (-1,)), self.indices_concatenated)
        return result


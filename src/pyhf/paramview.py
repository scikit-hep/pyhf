from . import get_backend, default_backend, events

def index_helper(name, indices, batch_shape, par_map, is_list):
    if isinstance(name,list):
        return [
            index_helper(x, indices, batch_shape, par_map, is_list) for x in name
        ]
    parfield_slice = tuple(slice(None,None) for x in batch_shape) + (par_map[name]['slice'],)
    indices = indices[parfield_slice]
    return default_backend.tolist(indices)

class ParamViewer(object):
    """
    Helper class to extract parameter data from possibly batched input
    """
    def __init__(self, tensor_shape, par_map, name, regular = True):
        self.tensor_shape = tensor_shape
        self.batch_shape = tensor_shape[:-1]
        x = list(range(int(default_backend.product(tensor_shape))))
        self.indices = default_backend.reshape(x,tensor_shape)
        self.par_map = par_map
        self.is_list = isinstance(name,list)
        self.index_selection = index_helper(name, self.indices, self.batch_shape, self.par_map, self.is_list)
        
        self.regular = regular

        self._precompute()
        events.subscribe('tensorlib_changed')(self._precompute)

    def _precompute(self):
        tensorlib, _ = get_backend()
        if self.regular:
            self.indices = tensorlib.astensor(self.index_selection, dtype = 'int')
            
    def __repr__(self):
        return '({} with [{}] batched: {})'.format(self.tensor_shape,' '.join(list(self.par_map.keys())), bool(self.batch_shape))
    
    def expand_indices(self, indices):
        """
        """
        pass


    def get_slice(self,tensor):
        """
        Returns:
            list of parameter slices:
                type when batched: list of (batchsize, slicesize,) tensors
                type when not batched: list of (slicesize, ) tensors
        """
        tensorlib, _ = get_backend()
        result = tensorlib.gather(
            tensorlib.reshape(tensor,(-1,)),
            self.indices
        )
        return result
   
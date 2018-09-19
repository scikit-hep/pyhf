from pyhf.interpolate import interpolator
import numpy as np
from .. import get_backend

def _prep_once(self,mtype):
    by_mname = {}
    for channel in self.spec['channels']:
        for sample in channel['samples']:
            for mname in sample['modifiers_by_type'].get(mtype,[]):
                cname = channel['name']
                sname = sample['name']
                by_mname.setdefault(mname,[]).append([cname,sname])
    
    all_mnames, all_affected_lists = list(zip(*by_mname.items()))
    all_histosets = []
    
    if mtype == 'normsys':
        interpcode = 1
    elif mtype == 'histosys':
        interpcode = 0
    else:
        raise RuntimeError('not sure')
    for mname, affected_list in zip(all_mnames,all_affected_lists):
        modifier = self.config.modifier(mname)
        histosets = []
        for cname,sname in affected_list:
            if mtype == 'normsys':
                nom = modifier.at_zero 
            elif mtype == 'histosys':
                nom = modifier.at_zero[cname][sname]
            else:
                raise RuntimeError('not sure')

            histoset = [
                modifier.at_minus_one[cname][sname],
                nom,
                modifier.at_plus_one[cname][sname]
            ]
            
                    
            
            histosets.append(histoset)
        all_histosets.append(histosets)

    fh = filled_shapes_histos(all_mnames,all_affected_lists,all_histosets)

    slice_map = {mname: self.config.par_slice(mname) for mname in all_mnames}
    
    return (all_mnames,all_affected_lists,all_histosets),fh,slice_map,interpcode

def _prep_pars(slice_map,all_mnames,pars):
    all_parslices = [pars[slice_map[mname]] for mname in all_mnames]
    return all_parslices

def generate_shapes_alphas(alphasets):
    a_shape = (len(alphasets),max(map(len,alphasets)))
    return tuple(a_shape)

def generate_shapes_histos(histogramssets):
    h_shape = [len(histogramssets),0,0,0]
    r_shape = [len(histogramssets),0,0]
        
    for hs in histogramssets:
        h_shape[1] = max(h_shape[1],len(hs))
        for h in hs:
            h_shape[2] = max(h_shape[2],len(h))
            for sh in h:
                h_shape[3] = max(h_shape[3],len(sh))
    return tuple(h_shape),tuple(h_shape[:-2]+[1]+h_shape[-1:])

def filled_shapes_alpha(alphasets):
    tensorlib, _ = get_backend()
    alphas = generate_shapes_alphas(alphasets)
    alphas = tensorlib.ones(alphas)*np.nan
    for i,alphaset in enumerate(alphasets):
        alphas[i,:len(alphaset)] = alphaset
    return alphas

def filled_shapes_histos(mnames,afnames,histogramssets):
    tensorlib, _ = get_backend()
    histos, result = generate_shapes_histos(histogramssets)
    histos,result = tensorlib.ones(histos)*np.nan, tensorlib.ones(result)*np.nan,
    histo_num, histomap = 0,{}
    for i,(mname,af,syst) in enumerate(zip(mnames,afnames,histogramssets)):
        for j,((cname,sname),sample) in enumerate(zip(af,syst)):
            for k,variation in enumerate(sample):
                histos[i,j,k,:len(variation)] = variation
            result[i,j,0,:len(sample[0])] = tensorlib.ones(len(sample[0]))*histo_num
            histomap.setdefault(mname,{}).setdefault(cname,{})[sname] = histo_num
            histo_num +=1
    mask_map = {num: (result.ravel()==num) for num in range(histo_num)}
    return histos, histomap, result, mask_map

def make_slice_map(mask_map):
    slices = {}
    ##the histogram are
    ##consecutive in the 
    ##result, so we can make slices
    for n,v in mask_map.items():
        indices = np.where(v) 
        len(indices)
        first,last = None,None
        ison = False
        final = False
        for i,k in enumerate(v):
            if k and (not final):
                first = i
                ison = True
            elif (not k) and ison:
                last = i
                ison = False
                final = True
            elif k and final:
                raise RuntimeError('ok')
        slices[n] = slice(first,last)
    return slices

class CombinedInterpolator(object):
    def __init__(self,pdf,mtype):
        self.ponce = _prep_once(pdf,mtype)
        self.slice_map = make_slice_map(self.ponce[1][3])

    def _extract(self,results):
        return {k: results[v] for k,v in self.slice_map.items()}

    def _transform(self, split_results):
        ponce = self.ponce
        all_mnames,all_affected_lists, name_map, _, _ = ponce[0][0],ponce[0][1],ponce[1][1],ponce[1][2],ponce[1][3]
        mtype_results = {}
        
        for mname, affected_list in zip(all_mnames,all_affected_lists):
            for cname,sname in affected_list:
                num = name_map[mname][cname][sname]
                this_result = split_results[num]
                mtype_results.setdefault(cname,{}).setdefault(sname,[]).append(this_result)
        return mtype_results

    def __set_results(self, results):
        split_results = self._extract(results.ravel())
        mtype_results = self._transform(split_results)
        return mtype_results

    def _make_results(self,testpars):
        interpcode = self.ponce[3]
        peach = _prep_pars(self.ponce[2],self.ponce[0][0],testpars)
        fa    = filled_shapes_alpha(peach)
        return interpolator(interpcode)(self.ponce[1][0],fa)
   
    def apply(self,testpars):
        r = self._make_results(testpars)
        new_result = self.__set_results(r)
        return new_result
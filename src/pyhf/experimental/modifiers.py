from __future__ import annotations
import pyhf
from pyhf.parameters import ParamViewer
from pyhf import get_backend
from pyhf import events

from typing import Sequence, Callable, Any


class BaseApplier:
    ...


class BaseBuilder:
    ...


def _allocate_new_param(
    p: dict[str, Sequence[float]]
) -> dict[str, str | bool | int | Sequence[float]]:
    return {
        'paramset_type': 'unconstrained',
        'n_parameters': 1,
        'is_shared': True,
        'inits': p['inits'],
        'bounds': p['bounds'],
        'is_scalar': True,
        'fixed': False,
    }


def make_func(expression: str, deps: list[str]) -> Callable[[Sequence[float]], Any]:
    def func(d: Sequence[float]) -> Any:
        import numexpr as ne

        return ne.evaluate(expression, local_dict=dict(zip(deps, d)))

    return func


def make_builder(
    funcname: str, deps: list[str], newparams: dict[str, dict[str, Sequence[float]]]
) -> BaseBuilder:
    class _builder(BaseBuilder):
        def __init__(self, config):
            self.builder_data = {'funcs': {}}
            self.config = config

        def collect(self, thismod, nom):
            maskval = True if thismod else False
            mask = [maskval] * len(nom)
            return {'mask': mask}

        def append(self, key, channel, sample, thismod, defined_samp):
            self.builder_data.setdefault(key, {}).setdefault(sample, {}).setdefault(
                'data', {'mask': []}
            )
            nom = (
                defined_samp['data']
                if defined_samp
                else [0.0] * self.config.channel_nbins[channel]
            )
            moddata = self.collect(thismod, nom)
            self.builder_data[key][sample]['data']['mask'] += moddata['mask']
            if thismod:
                if thismod['name'] != funcname:
                    print(thismod)
                    self.builder_data['funcs'].setdefault(
                        thismod['name'], thismod['data']['expr']
                    )
                self.required_parsets = {
                    k: [_allocate_new_param(v)] for k, v in newparams.items()
                }

        def finalize(self):
            return self.builder_data

    return _builder


def make_applier(
    funcname: str, deps: list[str], newparams: dict[str, dict[str, Sequence[float]]]
) -> BaseApplier:
    class _applier(BaseApplier):
        name = funcname
        op_code = 'multiplication'

        def __init__(self, modifiers, pdfconfig, builder_data, batch_size=None):
            self.funcs = [make_func(v, deps) for v in builder_data['funcs'].values()]

            self.batch_size = batch_size
            pars_for_applier = deps
            _modnames = [f'{mtype}/{m}' for m, mtype in modifiers]

            parfield_shape = (
                (self.batch_size, pdfconfig.npars)
                if self.batch_size
                else (pdfconfig.npars,)
            )
            self.param_viewer = ParamViewer(
                parfield_shape, pdfconfig.par_map, pars_for_applier
            )
            self._custommod_mask = [
                [[builder_data[modname][s]['data']['mask']] for s in pdfconfig.samples]
                for modname in _modnames
            ]
            self._precompute()
            events.subscribe('tensorlib_changed')(self._precompute)

        def _precompute(self):
            tensorlib, _ = get_backend()
            if not self.param_viewer.index_selection:
                return
            self.custommod_mask = tensorlib.tile(
                tensorlib.astensor(self._custommod_mask),
                (1, 1, self.batch_size or 1, 1),
            )
            self.custommod_mask_bool = tensorlib.astensor(
                self.custommod_mask, dtype="bool"
            )
            self.custommod_default = tensorlib.ones(self.custommod_mask.shape)

        def apply(self, pars):
            """
            Returns:
                modification tensor: Shape (n_modifiers, n_global_samples, n_alphas, n_global_bin)
            """
            if not self.param_viewer.index_selection:
                return
            tensorlib, _ = get_backend()
            if self.batch_size is None:
                deps = self.param_viewer.get(pars)
                print('deps', deps.shape)
                results = tensorlib.astensor([f(deps) for f in self.funcs])
                results = tensorlib.einsum('msab,m->msab', self.custommod_mask, results)
            else:
                deps = self.param_viewer.get(pars)
                print('deps', deps.shape)
                results = tensorlib.astensor([f(deps) for f in self.funcs])
                results = tensorlib.einsum(
                    'msab,ma->msab', self.custommod_mask, results
                )
            results = tensorlib.where(
                self.custommod_mask_bool, results, self.custommod_default
            )
            return results

    return _applier


def add_custom_modifier(
    funcname: str, deps: list[str], newparams: dict[str, dict[str, Sequence[float]]]
) -> dict[str, tuple[BaseBuilder, BaseApplier]]:

    _builder = make_builder(funcname, deps, newparams)
    _applier = make_applier(funcname, deps, newparams)

    modifier_set = {_applier.name: (_builder, _applier)}
    modifier_set.update(**pyhf.modifiers.histfactory_set)
    return modifier_set

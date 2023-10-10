from __future__ import annotations

import logging
from typing import Any, Callable, Sequence

import pyhf
from pyhf import events, get_backend
from pyhf.parameters import ParamViewer

log = logging.getLogger(__name__)

__all__ = ["add_custom_modifier"]


def __dir__():
    return __all__


try:
    import numexpr as ne
except ModuleNotFoundError:
    log.error(
        "\nInstallation of the experimental extra is required to use pyhf.experimental.modifiers"
        + "\nPlease install with: python -m pip install 'pyhf[experimental]'\n",
        exc_info=True,
    )
    raise


class BaseApplier:
    ...


class BaseBuilder:
    ...


def _allocate_new_param(
    p: dict[str, Sequence[float]]
) -> dict[str, str | bool | int | Sequence[float]]:
    return {
        "paramset_type": "unconstrained",
        "n_parameters": 1,
        "is_shared": True,
        "inits": p["inits"],
        "bounds": p["bounds"],
        "is_scalar": True,
        "fixed": False,
    }


def make_func(expression: str, deps: list[str]) -> Callable[[Sequence[float]], Any]:
    def func(d: Sequence[float]) -> Any:
        return ne.evaluate(expression, local_dict=dict(zip(deps, d)))

    return func


def make_builder(
    func_name: str, deps: list[str], new_params: dict[str, dict[str, Sequence[float]]]
) -> BaseBuilder:
    class _builder(BaseBuilder):
        is_shared = False

        def __init__(self, config):
            self.builder_data = {"funcs": {}}
            self.config = config

        def collect(self, thismod, nom):
            maskval = bool(thismod)
            mask = [maskval] * len(nom)
            return {"mask": mask}

        def append(self, key, channel, sample, thismod, defined_sample):
            self.builder_data.setdefault(key, {}).setdefault(sample, {}).setdefault(
                "data", {"mask": []}
            )
            nom = (
                defined_sample["data"]
                if defined_sample
                else [0.0] * self.config.channel_nbins[channel]
            )
            mod_data = self.collect(thismod, nom)
            self.builder_data[key][sample]["data"]["mask"] += mod_data["mask"]
            if thismod:
                if thismod["name"] != func_name:
                    print(thismod)
                    self.builder_data["funcs"].setdefault(
                        thismod["name"], thismod["data"]["expr"]
                    )
                self.required_parsets = {
                    k: [_allocate_new_param(v)] for k, v in new_params.items()
                }

        def finalize(self):
            return self.builder_data

    return _builder


def make_applier(
    func_name: str, deps: list[str], new_params: dict[str, dict[str, Sequence[float]]]
) -> BaseApplier:
    class _applier(BaseApplier):
        name = func_name
        op_code = "multiplication"

        def __init__(self, modifiers, pdfconfig, builder_data, batch_size=None):
            self.funcs = [make_func(v, deps) for v in builder_data["funcs"].values()]

            self.batch_size = batch_size
            pars_for_applier = deps
            _mod_names = [f"{mtype}/{m}" for m, mtype in modifiers]

            parfield_shape = (
                (self.batch_size, pdfconfig.npars)
                if self.batch_size
                else (pdfconfig.npars,)
            )
            self.param_viewer = ParamViewer(
                parfield_shape, pdfconfig.par_map, pars_for_applier
            )
            self._custom_mod_mask = [
                [[builder_data[mod_name][s]["data"]["mask"]] for s in pdfconfig.samples]
                for mod_name in _mod_names
            ]
            self._precompute()
            events.subscribe("tensorlib_changed")(self._precompute)

        def _precompute(self):
            tensorlib, _ = get_backend()
            if not self.param_viewer.index_selection:
                return
            self.custom_mod_mask = tensorlib.tile(
                tensorlib.astensor(self._custom_mod_mask),
                (1, 1, self.batch_size or 1, 1),
            )
            self.custom_mod_mask_bool = tensorlib.astensor(
                self.custom_mod_mask, dtype="bool"
            )
            self.custom_mod_default = tensorlib.ones(self.custom_mod_mask.shape)

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
                print("deps", deps.shape)
                results = tensorlib.astensor([f(deps) for f in self.funcs])
                results = tensorlib.einsum(
                    "msab,m->msab", self.custom_mod_mask, results
                )
            else:
                deps = self.param_viewer.get(pars)
                print("deps", deps.shape)
                results = tensorlib.astensor([f(deps) for f in self.funcs])
                results = tensorlib.einsum(
                    "msab,ma->msab", self.custom_mod_mask, results
                )
            results = tensorlib.where(
                self.custom_mod_mask_bool, results, self.custom_mod_default
            )
            return results

    return _applier


def add_custom_modifier(
    func_name: str, deps: list[str], new_params: dict[str, dict[str, Sequence[float]]]
) -> dict[str, tuple[BaseBuilder, BaseApplier]]:
    r"""
    Add a custom modifier type with the modifier data defined through a custom
    numexpr string expression.

    Example:

        >>> import pyhf
        >>> import pyhf.experimental.modifiers
        >>> pyhf.set_backend("numpy")
        >>> new_params = {
        ...     "m1": {"inits": (1.0,), "bounds": ((-5.0, 5.0),)},
        ...     "m2": {"inits": (1.0,), "bounds": ((-5.0, 5.0),)},
        ... }
        >>> expanded_pyhf = pyhf.experimental.modifiers.add_custom_modifier(
        ...     "custom", ["m1", "m2"], new_params
        ... )
        >>> model = pyhf.Model(
        ...     {
        ...         "channels": [
        ...             {
        ...                 "name": "singlechannel",
        ...                 "samples": [
        ...                     {
        ...                         "name": "signal",
        ...                         "data": [10, 20],
        ...                         "modifiers": [
        ...                             {
        ...                                 "name": "f2",
        ...                                 "type": "custom",
        ...                                 "data": {"expr": "m1"},
        ...                             },
        ...                         ],
        ...                     },
        ...                     {
        ...                         "name": "background",
        ...                         "data": [100, 150],
        ...                         "modifiers": [
        ...                             {
        ...                                 "name": "f1",
        ...                                 "type": "custom",
        ...                                 "data": {"expr": "m1+(m2**2)"},
        ...                             },
        ...                         ],
        ...                     },
        ...                 ],
        ...             }
        ...         ]
        ...     },
        ...     modifier_set=expanded_pyhf,
        ...     poi_name="m1",
        ...     validate=False,
        ...     batch_size=1,
        ... )
        >>> model.config.modifiers
        [('f1', 'custom'), ('f2', 'custom')]

    Args:
        func_name (:obj:`str`): The name of the custom modifier type.
        deps (:obj:`list`): The names of the new parameters of the modifier
         function.
        new_params (:obj:`dict`): The new parameters.

    Returns:
        :obj:`dict`: The updated ``pyhf.modifiers.histfactory_set`` with the added
        custom modifier type.
    """
    _builder = make_builder(func_name, deps, new_params)
    _applier = make_applier(func_name, deps, new_params)

    modifier_set = {_applier.name: (_builder, _applier)}
    modifier_set.update(**pyhf.modifiers.histfactory_set)
    return modifier_set

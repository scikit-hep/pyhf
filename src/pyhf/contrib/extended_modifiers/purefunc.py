import jax
import jax.numpy as jnp
import sympy
import sympy.parsing.sympy_parser as parser

from pyhf.parameters import ParamViewer


class InvalidLanguage(Exception):
    """
    InvalidLanguage is raised when a specified language parsing an expression is declared.
    """


class InvalidExpression(Exception):
    """
    InvalidExpression is raised when the expressions cannot be fully resolved.
    """


def create_modifiers():

    class PureFunctionModifierBuilder:
        is_shared = True

        def __init__(self, pdfconfig, transforms):
            self.config = pdfconfig
            self.transforms = transforms
            self.required_parsets = {}
            self.builder_data = {"local": {}, "global": {"symbols": set()}}
            self.parsed_expressions = {}
            self.languages = {}
            self.free_symbols = set()
            self.parse_expressions()

        def parse_expressions(self):
            # Collect all bindings (names), expressions, and the parser language
            bindings = [_["names"] for _ in self.transforms]
            expressions = [_["expression"] for _ in self.transforms]
            languages = [_["language"] for _ in self.transforms]
            free_symbols = set()

            for bind, exp, lang in zip(bindings, expressions, languages):
                if lang != "sympy":
                    msg = f"Parser {lang} is not implemented."
                    raise InvalidLanguage(msg)
                parsed = parser.parse_expr(exp)

                if len(bind) > 1:
                    # TODO check that bind and parsed have same length
                    # Additional loop over tuple of expressions
                    for c, b in enumerate(bind):
                        sub_exp = parsed[c]
                        self.parsed_expressions[b] = sub_exp
                        self.languages[b] = lang
                else:
                    self.parsed_expressions[bind[0]] = parsed
                    self.languages[bind[0]] = lang

            # walk through all bindings-expressions pairs and substitute
            # bindings used as free symbols
            # throw an exception if the substitution is across languages (for future proofing)
            for binding, exp in self.parsed_expressions.items():
                symbols = exp.free_symbols
                for symb in symbols:
                    if str(symb) in self.parsed_expressions:
                        if self.languages[binding] != self.languages[str(symb)]:
                            msg = f"{binding} and {symb} must be parsed in the same language."
                            raise InvalidLanguage(msg)
                        exp = exp.subs(symb, self.parsed_expressions[str(symb)])  # noqa: PLW2901
                if self.languages[binding] == "sympy":
                    exp = sympy.simplify(exp)  # noqa: PLW2901
                self.parsed_expressions[binding] = exp
                free_symbols.update(exp.free_symbols)

            # check if any bindings remain in free_symbols, if so we have some cyclic dependency
            # and should throw an exception
            for symbol in free_symbols:
                self.builder_data["global"].setdefault("symbols", set()).add(symbol)
                if str(symbol) in self.parsed_expressions:
                    msg = f"{symbol} remains unresolved, you should investigate the expressions."
                    raise InvalidExpression(msg)
            list_of_symbols = [str(x) for x in free_symbols]
            self.required_parsets = self.require_symbols_as_scalars(list_of_symbols)
            self.builder_data["global"]["symbol_names"] = list_of_symbols

        def collect(self, thismod, nom):
            maskval = bool(thismod)
            mask = [maskval] * len(nom)
            return {"mask": mask}

        def require_symbols_as_scalars(self, symbols):
            return {
                p: [
                    {
                        "paramset_type": "unconstrained",
                        "n_parameters": 1,
                        "is_shared": True,
                        "inits": (1.0,),
                        "bounds": ((0, 10),),
                        "is_scalar": True,
                        "fixed": False,
                    }
                ]
                for p in symbols
            }

        def append(self, key, channel, sample, thismod, defined_samp):
            self.builder_data["local"].setdefault(key, {}).setdefault(
                sample, {}
            ).setdefault("data", {"mask": []})

            nom = (
                defined_samp["data"]
                if defined_samp
                else [0.0] * self.config.channel_nbins[channel]
            )
            moddata = self.collect(thismod, nom)
            self.builder_data["local"][key][sample]["data"]["mask"] += moddata["mask"]
            if thismod is not None:
                binding = thismod.get("name", None)
                expr = self.parsed_expressions[binding]
            else:
                expr = None
            self.builder_data["local"].setdefault(key, {}).setdefault(
                sample, {}
            ).setdefault("channels", {}).setdefault(channel, {})["parsed"] = expr

        def finalize(self):
            list_of_symbols = self.builder_data["global"]["symbol_names"]
            for _modname, modspec in self.builder_data["local"].items():  # noqa: PERF102
                for _sample, samplespec in modspec.items():  # noqa: PERF102
                    for _channel, channelspec in samplespec["channels"].items():  # noqa: PERF102
                        if channelspec["parsed"] is not None:
                            channelspec["jaxfunc"] = sympy.lambdify(
                                list_of_symbols, channelspec["parsed"], "jax"
                            )
                        else:
                            channelspec["jaxfunc"] = lambda *args: 1.0  # noqa: ARG005
            return self.builder_data

    class PureFunctionModifierApplicator:
        op_code = "multiplication"
        name = "purefunc"

        def __init__(
            self, modifiers=None, pdfconfig=None, builder_data=None, batch_size=None
        ):
            self.builder_data = builder_data
            self.batch_size = batch_size
            self.pdfconfig = pdfconfig
            self.inputs = [str(x) for x in builder_data["global"]["symbols"]]

            self.keys = [f"{mtype}/{m}" for m, mtype in modifiers]
            self.modifiers = [m for m, _ in modifiers]

            parfield_shape = (
                (self.batch_size, pdfconfig.npars)
                if self.batch_size
                else (pdfconfig.npars,)
            )

            self.param_viewer = ParamViewer(
                parfield_shape, pdfconfig.par_map, self.inputs
            )
            self.create_jax_eval()

        def create_jax_eval(self):
            def eval_func(pars):
                return jnp.array(
                    [
                        [
                            jnp.concatenate(
                                [
                                    self.builder_data["local"][m][s]["channels"][c][
                                        "jaxfunc"
                                    ](*pars)
                                    * jnp.ones(self.pdfconfig.channel_nbins[c])
                                    for c in self.pdfconfig.channels
                                ]
                            )
                            for s in self.pdfconfig.samples
                        ]
                        for m in self.keys
                    ]
                )

            self.jaxeval = eval_func

        def apply_nonbatched(self, pars):
            return jnp.expand_dims(self.jaxeval(pars), 2)

        def apply_batched(self, pars):
            return jax.vmap(self.jaxeval, in_axes=(1,), out_axes=2)(pars)

        def apply(self, pars):
            if not self.param_viewer.index_selection:
                return None
            if self.batch_size is None:
                par_selection = self.param_viewer.get(pars)
                results_purefunc = self.apply_nonbatched(par_selection)
            else:
                par_selection = self.param_viewer.get(pars)
                results_purefunc = self.apply_batched(par_selection)
            return results_purefunc

    return PureFunctionModifierBuilder, PureFunctionModifierApplicator


from pyhf.modifiers import histfactory_set


def enable():
    modifier_set = {}
    modifier_set.update(**histfactory_set)

    builder, applicator = create_modifiers()

    modifier_set.update(**{applicator.name: (builder, applicator)})
    return modifier_set

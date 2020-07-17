"""Numpy Backend Function Shim."""

from .. import get_backend


def make_func(
    objective,
    data,
    pdf,
    tv,
    fixed_values_tensor,
    fixed_idx=[],
    variable_idx=[],
    do_grad=False,
):
    tensorlib, _ = get_backend()

    assert not do_grad, "Numpy does not support autodifferentiation"

    def func(pars):
        pars = tensorlib.astensor(pars)
        constrained_pars = tv.stitch([fixed_values_tensor, pars])
        return objective(constrained_pars, data, pdf)

    return func

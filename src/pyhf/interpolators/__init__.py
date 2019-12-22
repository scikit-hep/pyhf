"""Histogram Interpolation."""


def _slow_interpolator_looper(histogramssets, alphasets, func):
    all_results = []
    for histoset, alphaset in zip(histogramssets, alphasets):
        all_results.append([])
        set_result = all_results[-1]
        for histo in histoset:
            set_result.append([])
            histo_result = set_result[-1]
            for alpha in alphaset:
                alpha_result = []
                for down, nom, up in zip(histo[0], histo[1], histo[2]):
                    v = func(down, nom, up, alpha)
                    alpha_result.append(v)
                histo_result.append(alpha_result)
    return all_results


# interpolation codes come from https://cds.cern.ch/record/1456844/files/CERN-OPEN-2012-016.pdf
from .code0 import code0, _slow_code0
from .code1 import code1, _slow_code1
from .code2 import code2, _slow_code2
from .code4 import code4, _slow_code4
from .code4p import code4p, _slow_code4p
from .. import exceptions


def get(interpcode, do_tensorized_calc=True):
    interpcodes = {
        0: code0 if do_tensorized_calc else _slow_code0,
        1: code1 if do_tensorized_calc else _slow_code1,
        2: code2 if do_tensorized_calc else _slow_code2,
        4: code4 if do_tensorized_calc else _slow_code4,
        '4p': code4p if do_tensorized_calc else _slow_code4p,
    }

    try:
        return interpcodes[interpcode]
    except KeyError:
        raise exceptions.InvalidInterpCode


__all__ = ['code0', 'code1', 'code2', 'code4', 'code4p']

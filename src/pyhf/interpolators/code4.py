"""Polynomial Interpolation (Code 4)."""
import logging
import math
from .. import get_backend, default_backend
from .. import events
from . import _slow_interpolator_looper

log = logging.getLogger(__name__)


class code4(object):
    r"""
    The polynomial interpolation and exponential extrapolation strategy.

    .. math::
        \sigma_{sb} (\vec{\alpha}) = \sigma_{sb}^0(\vec{\alpha}) \underbrace{\prod_{p \in \text{Syst}} I_\text{poly|exp.} (\alpha_p; \sigma_{sb}^0, \sigma_{psb}^+, \sigma_{psb}^-, \alpha_0)}_\text{factors to calculate}


    with

    .. math::
        I_\text{poly|exp.}(\alpha; I^0, I^+, I^-, \alpha_0) = \begin{cases} \left(\frac{I^+}{I^0}\right)^{\alpha} \qquad \alpha \geq \alpha_0\\ 1 + \sum_{i=1}^6 a_i \alpha^i \qquad |\alpha| < \alpha_0 \\ \left(\frac{I^-}{I^0}\right)^{-\alpha} \qquad \alpha < -\alpha_0 \end{cases}

    and the :math:`a_i` are fixed by the boundary conditions

    .. math::
        \sigma_{sb}(\alpha=\pm\alpha_0), \left.\frac{\mathrm{d}\sigma_{sb}}{\mathrm{d}\alpha}\right|_{\alpha=\pm\alpha_0}, \mathrm{ and } \left.\frac{\mathrm{d}^2\sigma_{sb}}{\mathrm{d}\alpha^2}\right|_{\alpha=\pm\alpha_0}.

    Namely that :math:`\sigma_{sb}(\vec{\alpha})` is continuous, and its first- and second-order derivatives are continuous as well.

    """

    def __init__(self, histogramssets, subscribe=True, alpha0=1):
        """Polynomial Interpolation."""
        # alpha0 is assumed to be positive and non-zero. If alpha0 == 0, then
        # we cannot calculate the coefficients (e.g. determinant == 0)
        assert alpha0 > 0
        self.__alpha0 = alpha0
        # nb: this should never be a tensor, store in default backend (e.g. numpy)
        self._histogramssets = default_backend.astensor(histogramssets)
        # initial shape will be (nsysts, 1)
        self.alphasets_shape = (self._histogramssets.shape[0], 1)
        # precompute terms that only depend on the histogramssets
        self._deltas_up = default_backend.divide(
            self._histogramssets[:, :, 2], self._histogramssets[:, :, 1]
        )
        self._deltas_dn = default_backend.divide(
            self._histogramssets[:, :, 0], self._histogramssets[:, :, 1]
        )
        self._broadcast_helper = default_backend.ones(
            default_backend.shape(self._deltas_up)
        )
        self._alpha0 = self._broadcast_helper * self.__alpha0

        deltas_up_alpha0 = default_backend.power(self._deltas_up, self._alpha0)
        deltas_dn_alpha0 = default_backend.power(self._deltas_dn, self._alpha0)
        # x = A^{-1} b
        A_inverse = default_backend.astensor(
            [
                [
                    15.0 / (16 * alpha0),
                    -15.0 / (16 * alpha0),
                    -7.0 / 16.0,
                    -7.0 / 16.0,
                    1.0 / 16 * alpha0,
                    -1.0 / 16.0 * alpha0,
                ],
                [
                    3.0 / (2 * math.pow(alpha0, 2)),
                    3.0 / (2 * math.pow(alpha0, 2)),
                    -9.0 / (16 * alpha0),
                    9.0 / (16 * alpha0),
                    1.0 / 16,
                    1.0 / 16,
                ],
                [
                    -5.0 / (8 * math.pow(alpha0, 3)),
                    5.0 / (8 * math.pow(alpha0, 3)),
                    5.0 / (8 * math.pow(alpha0, 2)),
                    5.0 / (8 * math.pow(alpha0, 2)),
                    -1.0 / (8 * alpha0),
                    1.0 / (8 * alpha0),
                ],
                [
                    3.0 / (-2 * math.pow(alpha0, 4)),
                    3.0 / (-2 * math.pow(alpha0, 4)),
                    -7.0 / (-8 * math.pow(alpha0, 3)),
                    7.0 / (-8 * math.pow(alpha0, 3)),
                    -1.0 / (8 * math.pow(alpha0, 2)),
                    -1.0 / (8 * math.pow(alpha0, 2)),
                ],
                [
                    3.0 / (16 * math.pow(alpha0, 5)),
                    -3.0 / (16 * math.pow(alpha0, 5)),
                    -3.0 / (16 * math.pow(alpha0, 4)),
                    -3.0 / (16 * math.pow(alpha0, 4)),
                    1.0 / (16 * math.pow(alpha0, 3)),
                    -1.0 / (16 * math.pow(alpha0, 3)),
                ],
                [
                    1.0 / (2 * math.pow(alpha0, 6)),
                    1.0 / (2 * math.pow(alpha0, 6)),
                    -5.0 / (16 * math.pow(alpha0, 5)),
                    5.0 / (16 * math.pow(alpha0, 5)),
                    1.0 / (16 * math.pow(alpha0, 4)),
                    1.0 / (16 * math.pow(alpha0, 4)),
                ],
            ]
        )
        b = default_backend.stack(
            [
                deltas_up_alpha0 - self._broadcast_helper,
                deltas_dn_alpha0 - self._broadcast_helper,
                default_backend.log(self._deltas_up) * deltas_up_alpha0,
                -default_backend.log(self._deltas_dn) * deltas_dn_alpha0,
                default_backend.power(default_backend.log(self._deltas_up), 2)
                * deltas_up_alpha0,
                default_backend.power(default_backend.log(self._deltas_dn), 2)
                * deltas_dn_alpha0,
            ]
        )
        self._coefficients = default_backend.einsum(
            'rc,shb,cshb->rshb', A_inverse, self._broadcast_helper, b
        )

        self._precompute()
        if subscribe:
            events.subscribe('tensorlib_changed')(self._precompute)

    def _precompute(self):
        tensorlib, _ = get_backend()
        self.deltas_up = tensorlib.astensor(self._deltas_up)
        self.deltas_dn = tensorlib.astensor(self._deltas_dn)
        self.broadcast_helper = tensorlib.astensor(self._broadcast_helper)
        self.alpha0 = tensorlib.astensor(self._alpha0)
        self.coefficients = tensorlib.astensor(self._coefficients)
        self.bases_up = tensorlib.einsum(
            'sa,shb->shab', tensorlib.ones(self.alphasets_shape), self.deltas_up
        )
        self.bases_dn = tensorlib.einsum(
            'sa,shb->shab', tensorlib.ones(self.alphasets_shape), self.deltas_dn
        )
        self.mask_on = tensorlib.ones(self.alphasets_shape)
        self.mask_off = tensorlib.zeros(self.alphasets_shape)
        self.ones = tensorlib.einsum(
            'sa,shb->shab', self.mask_on, self.broadcast_helper
        )

    def _precompute_alphasets(self, alphasets_shape):
        if alphasets_shape == self.alphasets_shape:
            return
        tensorlib, _ = get_backend()
        self.alphasets_shape = alphasets_shape
        self.bases_up = tensorlib.einsum(
            'sa,shb->shab', tensorlib.ones(self.alphasets_shape), self.deltas_up
        )
        self.bases_dn = tensorlib.einsum(
            'sa,shb->shab', tensorlib.ones(self.alphasets_shape), self.deltas_dn
        )
        self.mask_on = tensorlib.ones(self.alphasets_shape)
        self.mask_off = tensorlib.zeros(self.alphasets_shape)
        self.ones = tensorlib.einsum(
            'sa,shb->shab', self.mask_on, self.broadcast_helper
        )
        return

    def __call__(self, alphasets):
        """Compute Interpolated Values."""
        tensorlib, _ = get_backend()
        self._precompute_alphasets(tensorlib.shape(alphasets))

        # select where alpha >= alpha0 and produce the mask
        where_alphasets_gtalpha0 = tensorlib.where(
            alphasets >= self.__alpha0, self.mask_on, self.mask_off
        )
        masks_gtalpha0 = tensorlib.astensor(
            tensorlib.einsum(
                'sa,shb->shab', where_alphasets_gtalpha0, self.broadcast_helper
            ),
            dtype="bool",
        )

        # select where alpha > -alpha0 ["not(alpha <= -alpha0)"] and produce the mask
        where_alphasets_not_ltalpha0 = tensorlib.where(
            alphasets > -self.__alpha0, self.mask_on, self.mask_off
        )
        masks_not_ltalpha0 = tensorlib.astensor(
            tensorlib.einsum(
                'sa,shb->shab', where_alphasets_not_ltalpha0, self.broadcast_helper
            ),
            dtype="bool",
        )

        # s: set under consideration (i.e. the modifier)
        # a: alpha variation
        # h: histogram affected by modifier
        # b: bin of histogram
        exponents = tensorlib.einsum(
            'sa,shb->shab', tensorlib.abs(alphasets), self.broadcast_helper
        )
        # for |alpha| >= alpha0, we want to raise the bases to the exponent=alpha
        # and for |alpha| < alpha0, we want to raise the bases to the exponent=1
        masked_exponents = tensorlib.where(
            exponents >= self.__alpha0, exponents, self.ones
        )
        # we need to produce the terms of alpha^i for summing up
        alphasets_powers = tensorlib.stack(
            [
                alphasets,
                tensorlib.power(alphasets, 2),
                tensorlib.power(alphasets, 3),
                tensorlib.power(alphasets, 4),
                tensorlib.power(alphasets, 5),
                tensorlib.power(alphasets, 6),
            ]
        )
        # this is the 1 + sum_i a_i alpha^i
        value_btwn = tensorlib.ones(exponents.shape) + tensorlib.einsum(
            'rshb,rsa->shab', self.coefficients, alphasets_powers
        )

        # first, build a result where:
        #       alpha > alpha0   : fill with bases_up
        #   not(alpha > alpha0)  : fill with 1 + sum(a_i alpha^i)
        results_gtalpha0_btwn = tensorlib.where(
            masks_gtalpha0, self.bases_up, value_btwn
        )
        # then, build a result where:
        #      alpha >= -alpha0  : do nothing (fill with previous result)
        #   not(alpha >= -alpha0): fill with bases_dn
        bases = tensorlib.where(
            masks_not_ltalpha0, results_gtalpha0_btwn, self.bases_dn
        )
        return tensorlib.power(bases, masked_exponents)


class _slow_code4(object):
    """
    Reference Implementation of Code 4.

    delta_up^alpha0 = 1 + a1 alpha0 + a2 alpha0^2 + a3 alpha0^3 + a4 alpha0^4 + a5 alpha0^5 + a6 alpha0^6
    delta_down^alpha0 = 1 - a1 alpha0 + a2 alpha0^2 - a3 alpha0^3 + a4 alpha0^4 - a5 alpha0^5 + a6 alpha0^6

    f[alpha_] := 1 + a1 * alpha + a2 * alpha^2 + a3 * alpha^3 + a4 * alpha^4 + a5 * alpha^5 + a6 * alpha^6
    up[alpha_] := delta_up^alpha
    down[alpha_] := delta_down^(-alpha)

    We want to find the coefficients a1, a2, a3, a4, a5, a6 by solving:
      f[alpha0] == up[alpha0]
      f[-alpha0] == down[-alpha0]
      f'[alpha0] == up'[alpha0]
      f'[-alpha0] == down'[-alpha0]
      f''[alpha0] == up''[alpha0]
      f''[-alpha0] == down''[-alpha0]

    Treating this as multiplication with a rank-6 matrix A: A*x = b, where x = [a1, a2, a3, a4, a5, a6]

    [alpha0,  alpha0^2, alpha0^3,   alpha0^4,   alpha0^5,     alpha0^6  ] [a1] = [ delta_up^(alpha0) - 1                ]
    [-alpha0, alpha0^2, -alpha0^3,  alpha0^4,   -alpha0^5,    alpha0^6  ] [a2] = [ delta_down^(alpha0) - 1              ]
    [1,       2alpha0,  3alpha0^2,  4alpha0^3,  5alpha0^4,    6alpha0^5 ] [a3] = [ ln(delta_up) delta_up^(alpha0)       ]
    [1,       -2alpha0, 3alpha0^2,  -4alpha0^3, 5alpha0^4,    -6alpha0^5] [a4] = [ - ln(delta_down) delta_down^(alpha0) ]
    [0,       2,        6alpha0,    12alpha0^2, 20alpha0^3,   30alpha0^4] [a5] = [ ln(delta_up)^2 delta_up^(alpha0)     ]
    [0,       2,        -6alpha0,   12alpha0^2, -20alpha0^3,  30alpha0^4] [a6] = [ ln(delta_down)^2 delta_down^(alpha0) ]

    The determinant of this matrix is -2048*alpha0^15. The trace is 30*alpha0^4+16*alpha0^3+4*alpha0^2+alpha0. Therefore, this matrix is invertible if and only if alpha0 != 0.

    The inverse of this matrix is (and verifying with http://wims.unice.fr/~wims/wims.cgi)

    [15/(16*alpha0),  -15/(16*alpha0),  -7/16,            -7/16,            1/16*alpha0,      -1/16*alpha0    ]
    [3/(2*alpha0^2),  3/(2*alpha0^2),   -9/(16*alpha0),   9/(16*alpha0),    1/16,             1/16            ]
    [-5/(8*alpha0^3), 5/(8*alpha0^3),   5/(8*alpha0^2),   5/(8*alpha0^2),   -1/(8*alpha0),    1/(8*alpha0)    ]
    [3/(-2*alpha0^4), 3/(-2*alpha0^4),  -7/(-8*alpha0^3), 7/(-8*alpha0^3),  -1/(8*alpha0^2),  -1/(8*alpha0^2) ]
    [3/(16*alpha0^5), -3/(16*alpha0^5), -3/(16*alpha0^4), -3/(16*alpha0^4), 1/(16*alpha0^3),  -1/(16*alpha0^3)]
    [1/(2*alpha0^6),  1/(2*alpha0^6),   -5/(16*alpha0^5), 5/(16*alpha0^5),  1/(16*alpha0^4),  1/(16*alpha0^4) ]

    """

    def product(self, down, nom, up, alpha):
        delta_up = up / nom
        delta_down = down / nom
        if alpha >= self.alpha0:
            delta = math.pow(delta_up, alpha)
        elif -self.alpha0 < alpha < self.alpha0:
            delta_up_alpha0 = math.pow(delta_up, self.alpha0)
            delta_down_alpha0 = math.pow(delta_down, self.alpha0)
            b = [
                delta_up_alpha0 - 1,
                delta_down_alpha0 - 1,
                math.log(delta_up) * delta_up_alpha0,
                -math.log(delta_down) * delta_down_alpha0,
                math.pow(math.log(delta_up), 2) * delta_up_alpha0,
                math.pow(math.log(delta_down), 2) * delta_down_alpha0,
            ]
            A_inverse = [
                [
                    15.0 / (16 * self.alpha0),
                    -15.0 / (16 * self.alpha0),
                    -7.0 / 16.0,
                    -7.0 / 16.0,
                    1.0 / 16 * self.alpha0,
                    -1.0 / 16.0 * self.alpha0,
                ],
                [
                    3.0 / (2 * math.pow(self.alpha0, 2)),
                    3.0 / (2 * math.pow(self.alpha0, 2)),
                    -9.0 / (16 * self.alpha0),
                    9.0 / (16 * self.alpha0),
                    1.0 / 16,
                    1.0 / 16,
                ],
                [
                    -5.0 / (8 * math.pow(self.alpha0, 3)),
                    5.0 / (8 * math.pow(self.alpha0, 3)),
                    5.0 / (8 * math.pow(self.alpha0, 2)),
                    5.0 / (8 * math.pow(self.alpha0, 2)),
                    -1.0 / (8 * self.alpha0),
                    1.0 / (8 * self.alpha0),
                ],
                [
                    3.0 / (-2 * math.pow(self.alpha0, 4)),
                    3.0 / (-2 * math.pow(self.alpha0, 4)),
                    -7.0 / (-8 * math.pow(self.alpha0, 3)),
                    7.0 / (-8 * math.pow(self.alpha0, 3)),
                    -1.0 / (8 * math.pow(self.alpha0, 2)),
                    -1.0 / (8 * math.pow(self.alpha0, 2)),
                ],
                [
                    3.0 / (16 * math.pow(self.alpha0, 5)),
                    -3.0 / (16 * math.pow(self.alpha0, 5)),
                    -3.0 / (16 * math.pow(self.alpha0, 4)),
                    -3.0 / (16 * math.pow(self.alpha0, 4)),
                    1.0 / (16 * math.pow(self.alpha0, 3)),
                    -1.0 / (16 * math.pow(self.alpha0, 3)),
                ],
                [
                    1.0 / (2 * math.pow(self.alpha0, 6)),
                    1.0 / (2 * math.pow(self.alpha0, 6)),
                    -5.0 / (16 * math.pow(self.alpha0, 5)),
                    5.0 / (16 * math.pow(self.alpha0, 5)),
                    1.0 / (16 * math.pow(self.alpha0, 4)),
                    1.0 / (16 * math.pow(self.alpha0, 4)),
                ],
            ]

            coefficients = [
                sum([A_i * b_j for A_i, b_j in zip(A_row, b)]) for A_row in A_inverse
            ]
            delta = 1
            for i in range(1, 7):
                delta += coefficients[i - 1] * math.pow(alpha, i)
        else:
            delta = math.pow(delta_down, (-alpha))
        return delta

    def __init__(self, histogramssets, subscribe=True, alpha0=1):
        self._histogramssets = histogramssets
        self.alpha0 = alpha0

    def __call__(self, alphasets):
        tensorlib, _ = get_backend()
        return tensorlib.astensor(
            _slow_interpolator_looper(
                self._histogramssets, tensorlib.tolist(alphasets), self.product
            )
        )

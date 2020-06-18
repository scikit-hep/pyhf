"""Piecewise-Linear + Polynomial Interpolation (Code 4p)."""
import logging
from .. import get_backend, default_backend
from .. import events
from . import _slow_interpolator_looper

log = logging.getLogger(__name__)


class code4p(object):
    r"""
    The piecewise-linear interpolation strategy, with polynomial at :math:`\left|a\right| < 1`.

    .. math::
        \sigma_{sb} (\vec{\alpha}) = \sigma_{sb}^0(\vec{\alpha}) + \underbrace{\sum_{p \in \text{Syst}} I_\text{lin.} (\alpha_p; \sigma_{sb}^0, \sigma_{psb}^+, \sigma_{psb}^-)}_\text{deltas to calculate}

    """

    def __init__(self, histogramssets, subscribe=True):
        """Piecewise-Linear  + Polynomial Interpolation."""
        # nb: this should never be a tensor, store in default backend (e.g. numpy)
        self._histogramssets = default_backend.astensor(histogramssets)
        # initial shape will be (nsysts, 1)
        self.alphasets_shape = (self._histogramssets.shape[0], 1)
        # precompute terms that only depend on the histogramssets
        self._deltas_up = self._histogramssets[:, :, 2] - self._histogramssets[:, :, 1]
        self._deltas_dn = self._histogramssets[:, :, 1] - self._histogramssets[:, :, 0]
        self._broadcast_helper = default_backend.ones(
            default_backend.shape(self._deltas_up)
        )
        self._precompute()
        if subscribe:
            events.subscribe('tensorlib_changed')(self._precompute)

    def _precompute(self):
        tensorlib, _ = get_backend()
        self.deltas_up = tensorlib.astensor(self._deltas_up)
        self.deltas_dn = tensorlib.astensor(self._deltas_dn)

        self.S = 0.5 * (self.deltas_up + self.deltas_dn)
        self.A = 0.0625 * (self.deltas_up - self.deltas_dn)

        self.broadcast_helper = tensorlib.astensor(self._broadcast_helper)
        self.mask_on = tensorlib.ones(self.alphasets_shape)
        self.mask_off = tensorlib.zeros(self.alphasets_shape)

    def _precompute_alphasets(self, alphasets_shape):
        if alphasets_shape == self.alphasets_shape:
            return
        tensorlib, _ = get_backend()
        self.alphasets_shape = alphasets_shape
        self.mask_on = tensorlib.ones(self.alphasets_shape)
        self.mask_off = tensorlib.zeros(self.alphasets_shape)

    def __call__(self, alphasets):
        """Compute Interpolated Values."""
        tensorlib, _ = get_backend()
        self._precompute_alphasets(tensorlib.shape(alphasets))
        where_alphasets_greater_p1 = tensorlib.where(
            alphasets > 1, self.mask_on, self.mask_off
        )

        where_alphasets_smaller_m1 = tensorlib.where(
            alphasets < -1, self.mask_on, self.mask_off
        )

        # s: set under consideration (i.e. the modifier)
        # a: alpha variation
        # h: histogram affected by modifier
        # b: bin of histogram

        # for a > 1
        alphas_times_deltas_up = tensorlib.einsum(
            'sa,shb->shab', alphasets, self.deltas_up
        )

        # for a < -1
        alphas_times_deltas_dn = tensorlib.einsum(
            'sa,shb->shab', alphasets, self.deltas_dn
        )

        # for |a| < 1
        asquare = tensorlib.power(alphasets, 2)
        tmp1 = asquare * 3.0 - 10.0
        tmp2 = asquare * tmp1 + 15.0
        tmp3 = asquare * tmp2

        tmp3_times_A = tensorlib.einsum('sa,shb->shab', tmp3, self.A)

        alphas_times_S = tensorlib.einsum('sa,shb->shab', alphasets, self.S)

        deltas = tmp3_times_A + alphas_times_S
        # end |a| < 1

        masks_p1 = tensorlib.astensor(
            tensorlib.einsum(
                'sa,shb->shab', where_alphasets_greater_p1, self.broadcast_helper
            ),
            dtype='bool',
        )

        masks_m1 = tensorlib.astensor(
            tensorlib.einsum(
                'sa,shb->shab', where_alphasets_smaller_m1, self.broadcast_helper
            ),
            dtype='bool',
        )

        return tensorlib.where(
            masks_m1,
            alphas_times_deltas_dn,
            tensorlib.where(masks_p1, alphas_times_deltas_up, deltas),
        )


class _slow_code4p(object):
    def summand(self, down, nom, up, alpha):
        delta_up = up - nom
        delta_down = nom - down
        S = 0.5 * (delta_up + delta_down)
        A = 0.0625 * (delta_up - delta_down)
        if alpha > 1:
            delta = delta_up * alpha
        elif alpha < -1:
            delta = delta_down * alpha
        else:
            delta = alpha * (
                S + alpha * A * (15 + alpha * alpha * (-10 + alpha * alpha * 3))
            )
        return delta

    def __init__(self, histogramssets, subscribe=True):
        self._histogramssets = histogramssets

    def __call__(self, alphasets):
        tensorlib, _ = get_backend()
        return tensorlib.astensor(
            _slow_interpolator_looper(
                self._histogramssets, tensorlib.tolist(alphasets), self.summand
            )
        )

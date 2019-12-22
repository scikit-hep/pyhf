"""Piecewise-Exponential Interpolation (Code 1)."""
import logging
import math
from .. import get_backend, default_backend
from .. import events
from . import _slow_interpolator_looper

log = logging.getLogger(__name__)


class code1(object):
    r"""
    The piecewise-exponential interpolation strategy.

    .. math::
        \sigma_{sb} (\vec{\alpha}) = \sigma_{sb}^0(\vec{\alpha}) \underbrace{\prod_{p \in \text{Syst}} I_\text{exp.} (\alpha_p; \sigma_{sb}^0, \sigma_{psb}^+, \sigma_{psb}^-)}_\text{factors to calculate}


    with

    .. math::
        I_\text{exp.}(\alpha; I^0, I^+, I^-) = \begin{cases} \left(\frac{I^+}{I^0}\right)^{\alpha} \qquad \alpha \geq 0\\ \left(\frac{I^-}{I^0}\right)^{-\alpha} \qquad \alpha < 0 \end{cases}


    """

    def __init__(self, histogramssets, subscribe=True):
        """Piecewise-Exponential Interpolation."""
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

        self._precompute()
        if subscribe:
            events.subscribe('tensorlib_changed')(self._precompute)

    def _precompute(self):
        tensorlib, _ = get_backend()
        self.deltas_up = tensorlib.astensor(self._deltas_up)
        self.deltas_dn = tensorlib.astensor(self._deltas_dn)
        self.broadcast_helper = tensorlib.astensor(self._broadcast_helper)
        self.bases_up = tensorlib.einsum(
            'sa,shb->shab', tensorlib.ones(self.alphasets_shape), self.deltas_up
        )
        self.bases_dn = tensorlib.einsum(
            'sa,shb->shab', tensorlib.ones(self.alphasets_shape), self.deltas_dn
        )
        self.mask_on = tensorlib.ones(self.alphasets_shape)
        self.mask_off = tensorlib.zeros(self.alphasets_shape)

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
        return

    def __call__(self, alphasets):
        """Compute Interpolated Values."""
        tensorlib, _ = get_backend()
        self._precompute_alphasets(tensorlib.shape(alphasets))
        where_alphasets_positive = tensorlib.where(
            alphasets > 0, self.mask_on, self.mask_off
        )

        # s: set under consideration (i.e. the modifier)
        # a: alpha variation
        # h: histogram affected by modifier
        # b: bin of histogram
        exponents = tensorlib.einsum(
            'sa,shb->shab', tensorlib.abs(alphasets), self.broadcast_helper
        )
        masks = tensorlib.astensor(
            tensorlib.einsum(
                'sa,shb->shab', where_alphasets_positive, self.broadcast_helper
            ),
            dtype="bool",
        )

        bases = tensorlib.where(masks, self.bases_up, self.bases_dn)
        return tensorlib.power(bases, exponents)


class _slow_code1(object):
    def product(self, down, nom, up, alpha):
        delta_up = up / nom
        delta_down = down / nom
        if alpha > 0:
            delta = math.pow(delta_up, alpha)
        else:
            delta = math.pow(delta_down, (-alpha))
        return delta

    def __init__(self, histogramssets, subscribe=True):
        self._histogramssets = histogramssets

    def __call__(self, alphasets):
        tensorlib, _ = get_backend()
        return tensorlib.astensor(
            _slow_interpolator_looper(
                self._histogramssets, tensorlib.tolist(alphasets), self.product
            )
        )

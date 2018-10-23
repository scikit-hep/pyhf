import logging
from .. import get_backend, default_backend
from .. import events
from . import _slow_interpolator_looper

log = logging.getLogger(__name__)


class code2(object):
    r"""
    The quadratic interpolation and linear extrapolation strategy.

    .. math::
        \sigma_{sb} (\vec{\alpha}) = \sigma_{sb}^0(\vec{\alpha}) + \underbrace{\sum_{p \in \text{Syst}} I_\text{quad.|lin.} (\alpha_p; \sigma_{sb}^0, \sigma_{psb}^+, \sigma_{psb}^-)}_\text{deltas to calculate}


    with

    .. math::
        I_\text{quad.|lin.}(\alpha; I^0, I^+, I^-) = \begin{cases} (b + 2a)(\alpha - 1) \qquad \alpha \geq 1\\  a\alpha^2 + b\alpha \qquad |\alpha| < 1 \\ (b - 2a)(\alpha + 1) \qquad \alpha < -1 \end{cases}

    and

    .. math::
        a = \frac{1}{2} (I^+ + I^-) - I^0 \qquad \mathrm{and} \qquad b = \frac{1}{2}(I^+ - I^-)

    """

    def __init__(self, histogramssets, subscribe=True):
        # nb: this should never be a tensor, store in default backend (e.g. numpy)
        self._histogramssets = default_backend.astensor(histogramssets)
        # initial shape will be (nsysts, 1)
        self.alphasets_shape = (self._histogramssets.shape[0], 1)
        self._precompute()
        if subscribe:
            events.subscribe('tensorlib_changed')(self._precompute)

    def _precompute(self):
        tensorlib, _ = get_backend()
        self.a = tensorlib.astensor(
            0.5 * (self._histogramssets[:, :, 2] + self._histogramssets[:, :, 0])
            - self._histogramssets[:, :, 1]
        )
        self.b = tensorlib.astensor(
            0.5 * (self._histogramssets[:, :, 2] - self._histogramssets[:, :, 0])
        )
        self.b_plus_2a = self.b + 2 * self.a
        self.b_minus_2a = self.b - 2 * self.a
        # make up the masks correctly
        self.broadcast_helper = tensorlib.ones(tensorlib.shape(self.a))
        self.mask_on = tensorlib.ones(self.alphasets_shape)
        self.mask_off = tensorlib.zeros(self.alphasets_shape)
        self.ones = tensorlib.ones(self.alphasets_shape)

    def _precompute_alphasets(self, alphasets_shape):
        if alphasets_shape == self.alphasets_shape:
            return
        tensorlib, _ = get_backend()
        self.mask_on = tensorlib.ones(alphasets_shape)
        self.mask_off = tensorlib.zeros(alphasets_shape)
        self.ones = tensorlib.ones(alphasets_shape)
        self.alphasets_shape = alphasets_shape

    def __call__(self, alphasets):
        tensorlib, _ = get_backend()
        self._precompute_alphasets(tensorlib.shape(alphasets))

        # select where alpha > 1
        where_alphasets_gt1 = tensorlib.where(
            alphasets > 1, self.mask_on, self.mask_off
        )

        # select where alpha >= -1
        where_alphasets_not_lt1 = tensorlib.where(
            alphasets >= -1, self.mask_on, self.mask_off
        )

        # s: set under consideration (i.e. the modifier)
        # a: alpha variation
        # h: histogram affected by modifier
        # b: bin of histogram
        value_gt1 = tensorlib.einsum(
            'sa,shb->shab', alphasets - self.ones, self.b_plus_2a
        )
        value_btwn = tensorlib.einsum(
            'sa,sa,shb->shab', alphasets, alphasets, self.a
        ) + tensorlib.einsum('sa,shb->shab', alphasets, self.b)
        value_lt1 = tensorlib.einsum(
            'sa,shb->shab', alphasets + self.ones, self.b_minus_2a
        )

        masks_gt1 = tensorlib.einsum(
            'sa,shb->shab', where_alphasets_gt1, self.broadcast_helper
        )
        masks_not_lt1 = tensorlib.einsum(
            'sa,shb->shab', where_alphasets_not_lt1, self.broadcast_helper
        )

        # first, build a result where:
        #       alpha > 1   : fill with (b+2a)(alpha - 1)
        #   not(alpha > 1)  : fill with (a * alpha^2 + b * alpha)
        results_gt1_btwn = tensorlib.where(masks_gt1, value_gt1, value_btwn)
        # then, build a result where:
        #      alpha >= -1  : do nothing (fill with previous result)
        #   not(alpha >= -1): fill with (b-2a)(alpha + 1)
        return tensorlib.where(masks_not_lt1, results_gt1_btwn, value_lt1)


class _slow_code2(object):
    def summand(self, down, nom, up, alpha):
        a = 0.5 * (up + down) - nom
        b = 0.5 * (up - down)
        if alpha > 1:
            delta = (b + 2 * a) * (alpha - 1)
        elif -1 <= alpha <= 1:
            delta = a * alpha * alpha + b * alpha
        else:
            delta = (b - 2 * a) * (alpha + 1)
        return delta

    def __init__(self, histogramssets):
        self._histogramssets = histogramssets

    def __call__(self, alphasets):
        tensorlib, _ = get_backend()
        return tensorlib.astensor(
            _slow_interpolator_looper(
                self._histogramssets, tensorlib.tolist(alphasets), self.summand
            )
        )

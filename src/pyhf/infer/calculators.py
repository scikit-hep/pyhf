"""
Calculators for Hypothesis Testing.

The role of the calculators is to compute test statistic and
provide distributions of said test statistic under various
hypotheses.

Using the calculators hypothesis tests can then be performed.
"""
from .mle import fixed_poi_fit
from .. import get_backend
from .test_statistics import qmu, qmu_tilde
import tqdm


def create_calculator(calctype, *args, **kwargs):
    return {'asymptotics': AsymptoticCalculator, 'toybased': ToyCalculator,}[
        calctype
    ](*args, **kwargs)


def generate_asimov_data(asimov_mu, data, pdf, init_pars, par_bounds, fixed_params):
    """
    Compute Asimov Dataset (expected yields at best-fit values) for a given POI value.

    Args:
        asimov_mu (:obj:`float`): The value for the parameter of interest to be used.
        data (:obj:`tensor`): The observed data.
        pdf (~pyhf.pdf.Model): The statistical model adhering to the schema ``model.json``.
        init_pars (:obj:`tensor`): The initial parameter values to be used for fitting.
        par_bounds (:obj:`tensor`): The parameter value bounds to be used for fitting.
        fixed_params (:obj:`tensor`): Parameters to be held constant in the fit.

    Returns:
        Tensor: The Asimov dataset.

    """
    bestfit_nuisance_asimov = fixed_poi_fit(
        asimov_mu, data, pdf, init_pars, par_bounds, fixed_params
    )
    return pdf.expected_data(bestfit_nuisance_asimov)


class AsymptoticTestStatDistribution(object):
    r"""
    The distribution the test statistic in the asymptotic case.

    Note: These distributions are in :math:`-\hat{\mu}/\sigma` space.
    In the ROOT implementation the same sigma is assumed for both hypotheses
    and :math:`p`-values etc are computed in that space.
    This assumption is necessarily valid, but we keep this for compatibility reasons.

    In the :math:`-\hat{\mu}/\sigma` space, the test statistic (i.e. :math:`\hat{\mu}/\sigma`) is
    normally distributed with unit variance and its mean at
    the :math:`-\mu'`, where :math:`\mu'` is the true poi value of the hypothesis.
    """

    def __init__(self, shift):
        """
        Asymptotic test statistic distribution.

        Args:
            shift (:obj:`float`): The displacement of the test statistic distribution.

        Returns:
            ~pyhf.infer.calculators.AsymptoticTestStatDistribution: The asymptotic distribution of test statistic.

        """
        self.shift = shift
        self.sqrtqmuA_v = None

    def cdf(self, value):
        """
        Compute the value of the cumulative distribution function for a given value of the test statistic.

        Args:
            value (:obj:`float`): The test statistic value.

        Returns:
            Float: The integrated probability to observe a test statistic less than or equal to the observed ``value``.

        """
        tensorlib, _ = get_backend()
        return tensorlib.normal_cdf((value - self.shift))

    def pvalue(self, value):
        r"""
        The :math:`p`-value for a given value of the test statistic corresponding
        to signal strength :math:`\mu` and Asimov strength :math:`\mu'` as
        defined in Equations (59) and (57) of :xref:`arXiv:1007.1727`

        .. math::

            p_{\mu} = 1-F\left(q_{\mu}\middle|\mu'\right) = 1- \Phi\left(\sqrt{q_{\mu}} - \frac{\left(\mu-\mu'\right)}{\sigma}\right)

        with Equation (29)

        .. math::

            \frac{(\mu-\mu')}{\sigma} = \sqrt{\Lambda}= \sqrt{q_{\mu,A}}

        given the observed test statistics :math:`q_{\mu}` and :math:`q_{\mu,A}`.

        Args:
            value (:obj:`float`): The test statistic value.

        Returns:
            Float: The integrated probability to observe a value at least as large as the observed one.

        """
        tensorlib, _ = get_backend()
        # computing cdf(-x) instead of 1-cdf(x) for right-tail p-value for improved numerical stability
        return tensorlib.normal_cdf(-(value - self.shift))

    def expected_value(self, nsigma):
        """
        Return the expected value of the test statistic.

        Args:
            nsigma (:obj:`int` or :obj:`tensor`): The number of standard deviations.

        Returns:
            Float: The expected value of the test statistic.
        """
        return self.shift + nsigma


class AsymptoticCalculator(object):
    """The Asymptotic Calculator."""

    def __init__(
        self,
        data,
        pdf,
        init_pars=None,
        par_bounds=None,
        fixed_params=None,
        qtilde=True,
    ):
        """
        Asymptotic Calculator.

        Args:
            data (:obj:`tensor`): The observed data.
            pdf (~pyhf.pdf.Model): The statistical model adhering to the schema ``model.json``.
            init_pars (:obj:`tensor`): The initial parameter values to be used for fitting.
            par_bounds (:obj:`tensor`): The parameter value bounds to be used for fitting.
            fixed_params (:obj:`tensor`): Whether to fix the parameter to the init_pars value during minimization
            qtilde (:obj:`bool`): When ``True`` use :func:`~pyhf.infer.test_statistics.qmu_tilde`
             as the test statistic.
             When ``False`` use :func:`~pyhf.infer.test_statistics.qmu`.

        Returns:
            ~pyhf.infer.calculators.AsymptoticCalculator: The calculator for asymptotic quantities.

        """
        self.data = data
        self.pdf = pdf
        self.init_pars = init_pars or pdf.config.suggested_init()
        self.par_bounds = par_bounds or pdf.config.suggested_bounds()
        self.fixed_params = fixed_params or pdf.config.suggested_fixed()

        self.qtilde = qtilde
        self.sqrtqmuA_v = None

    def distributions(self, poi_test):
        """
        Probability distributions of the test statistic, as defined in
        :math:`\S` 3 of :xref:`arXiv:1007.1727` under the Wald approximation,
        under the signal + background and background-only hypotheses.

        Args:
            poi_test: The value for the parameter of interest.

        Returns:
            Tuple (~pyhf.infer.calculators.AsymptoticTestStatDistribution): The distributions under the hypotheses.

        """
        if self.sqrtqmuA_v is None:
            raise RuntimeError('need to call .teststatistic(poi_test) first')
        sb_dist = AsymptoticTestStatDistribution(-self.sqrtqmuA_v)
        b_dist = AsymptoticTestStatDistribution(0.0)
        return sb_dist, b_dist

    def teststatistic(self, poi_test):
        """
        Compute the test statistic for the observed data under the studied model.

        Args:
            poi_test: The value for the parameter of interest.

        Returns:
            Float: the value of the test statistic.

        """
        tensorlib, _ = get_backend()

        teststat_func = qmu_tilde if self.qtilde else qmu

        qmu_v = teststat_func(
            poi_test,
            self.data,
            self.pdf,
            self.init_pars,
            self.par_bounds,
            self.fixed_params,
        )
        sqrtqmu_v = tensorlib.sqrt(qmu_v)

        asimov_mu = 0.0
        asimov_data = generate_asimov_data(
            asimov_mu,
            self.data,
            self.pdf,
            self.init_pars,
            self.par_bounds,
            self.fixed_params,
        )
        qmuA_v = teststat_func(
            poi_test,
            asimov_data,
            self.pdf,
            self.init_pars,
            self.par_bounds,
            self.fixed_params,
        )
        self.sqrtqmuA_v = tensorlib.sqrt(qmuA_v)

        if not self.qtilde:  # qmu
            teststat = sqrtqmu_v - self.sqrtqmuA_v
        else:  # qtilde

            def _true_case():
                teststat = sqrtqmu_v - self.sqrtqmuA_v
                return teststat

            def _false_case():
                qmu = tensorlib.power(sqrtqmu_v, 2)
                qmu_A = tensorlib.power(self.sqrtqmuA_v, 2)
                teststat = (qmu - qmu_A) / (2 * self.sqrtqmuA_v)
                return teststat

            teststat = tensorlib.conditional(
                (sqrtqmu_v < self.sqrtqmuA_v), _true_case, _false_case
            )
        return teststat


class EmpiricalDistribution(object):
    def __init__(self, samples):
        self.samples = samples.ravel()

    def pvalue(self, value):
        tensorlib, _ = get_backend()
        return (
            tensorlib.where(self.samples >= value, 1, 0).sum()
            / tensorlib.shape(self.samples)[0]
        )

    def expected_value(self, nsigma):
        tensorlib, _ = get_backend()
        import numpy as np

        # TODO: tensorlib.percentile function
        return np.percentile(self.samples, (tensorlib.normal_cdf(nsigma)) * 100)


class ToyCalculator(object):
    def __init__(
        self,
        data,
        pdf,
        init_pars=None,
        par_bounds=None,
        qtilde=False,
        ntoys=2000,
        track_progress=True,
    ):
        self.ntoys = ntoys
        self.data = data
        self.pdf = pdf
        self.init_pars = init_pars or pdf.config.suggested_init()
        self.par_bounds = par_bounds or pdf.config.suggested_bounds()
        self.track_progress = track_progress

    def distributions(self, poi_test, track_progress=None):
        tensorlib, _ = get_backend()
        sample_shape = (self.ntoys,)

        signal_pars = self.pdf.config.suggested_init()
        signal_pars[self.pdf.config.poi_index] = poi_test
        signal_pdf = self.pdf.make_pdf(tensorlib.astensor(signal_pars))
        signal_sample = signal_pdf.sample(sample_shape)

        bkg_pars = self.pdf.config.suggested_init()
        bkg_pars[self.pdf.config.poi_index] = 0.0
        bkg_pdf = self.pdf.make_pdf(tensorlib.astensor(bkg_pars))
        bkg_sample = bkg_pdf.sample(sample_shape)

        tqdm_options = dict(
            total=self.ntoys,
            leave=False,
            disable=not (
                track_progress if track_progress is not None else self.track_progress
            ),
            unit='toy',
        )

        signal_qtilde = []
        for sample in tqdm.tqdm(signal_sample, **tqdm_options, desc='Signal-like'):
            signal_qtilde.append(
                qmu(poi_test, sample, self.pdf, signal_pars, self.par_bounds)
            )
        signal_qtilde = tensorlib.astensor(signal_qtilde)

        bkg_qtilde = []
        for sample in tqdm.tqdm(bkg_sample, **tqdm_options, desc='Background-like'):
            bkg_qtilde.append(
                qmu(poi_test, sample, self.pdf, bkg_pars, self.par_bounds)
            )
        bkg_qtilde = tensorlib.astensor(bkg_qtilde)

        s_plus_b = EmpiricalDistribution(signal_qtilde)
        b_only = EmpiricalDistribution(bkg_qtilde)
        return s_plus_b, b_only

    def teststatistic(self, poi_test):
        qmu_v = qmu(poi_test, self.data, self.pdf, self.init_pars, self.par_bounds)
        return qmu_v

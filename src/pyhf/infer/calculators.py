"""
Calculators for Hypothesis Testing.

The role of the calculators is to compute test statistic and
provide distributions of said test statistic under various
hypotheses.

Using the calculators hypothesis tests can then be performed.
"""
from .mle import fixed_poi_fit
from .. import get_backend
from .test_statistics import qmu
import tqdm


def create_calculator(calctype, *args, **kwargs):
    """
    Creates a calculator object of the specified `calctype`.

    See ~pyhf.infer.calculators.AsymptoticCalculator and ~pyhf.infer.calculators.ToyCalculator on additional arguments to be specified.

    Args:
        calctype (`str`): The calculator to create. Choose either 'asymptotics' or 'toybased'.

    Returns:
        calculator (`object`): A calculator.
    """
    return {'asymptotics': AsymptoticCalculator, 'toybased': ToyCalculator,}[calctype](
        *args, **kwargs
    )


def generate_asimov_data(asimov_mu, data, pdf, init_pars, par_bounds):
    """
    Compute Asimov Dataset (expected yields at best-fit values) for a given POI value.

    Args:
        asimov_mu (`float`): The value for the parameter of interest to be used.
        data (`tensor`): The observed data.
        pdf (~pyhf.pdf.Model): The statistical model adhering to the schema ``model.json``.
        init_pars (`tensor`): The initial parameter values to be used for fitting.
        par_bounds (`tensor`): The parameter value bounds to be used for fitting.

    Returns:
        Tensor: The Asimov dataset.

    """
    bestfit_nuisance_asimov = fixed_poi_fit(asimov_mu, data, pdf, init_pars, par_bounds)
    return pdf.expected_data(bestfit_nuisance_asimov)


class AsymptoticTestStatDistribution(object):
    """
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
            shift (`float`): The displacement of the test statistic distribution.

        Returns:
            ~pyhf.infer.calculators.AsymptoticTestStatDistribution: The asymptotic distribution of test statistic.

        """
        self.shift = shift
        self.sqrtqmuA_v = None

    def pvalue(self, value):
        """
        Compute the :math:`p`-value for a given value of the test statistic.

        Args:
            value (`float`): The test statistic value.

        Returns:
            Float: The integrated probability to observe a value at least as large as the observed one.

        """
        tensorlib, _ = get_backend()
        return 1 - tensorlib.normal_cdf(value - self.shift)

    def expected_value(self, nsigma):
        """
        Return the expected value of the test statistic.

        Args:
            nsigma (`int` or `tensor`): The number of standard deviations.

        Returns:
            Float: The expected value of the test statistic.
        """
        return nsigma


class AsymptoticCalculator(object):
    """The Asymptotic Calculator."""

    def __init__(self, data, pdf, init_pars=None, par_bounds=None, qtilde=False):
        """
        Asymptotic Calculator.

        Args:
            data (`tensor`): The observed data.
            pdf (~pyhf.pdf.Model): The statistical model adhering to the schema ``model.json``.
            init_pars (`tensor`): The initial parameter values to be used for fitting.
            par_bounds (`tensor`): The parameter value bounds to be used for fitting.
            qtilde (`bool`): Whether to use qtilde as the test statistic.

        Returns:
            ~pyhf.infer.calculators.AsymptoticCalculator: The calculator for asymptotic quantities.

        """
        self.data = data
        self.pdf = pdf
        self.init_pars = init_pars or pdf.config.suggested_init()
        self.par_bounds = par_bounds or pdf.config.suggested_bounds()
        self.qtilde = qtilde
        self.sqrtqmuA_v = None

    def distributions(self, poi_test):
        """
        Probability Distributions of the test statistic value under the signal + background and background-only hypothesis.

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
        qmu_v = qmu(poi_test, self.data, self.pdf, self.init_pars, self.par_bounds)
        sqrtqmu_v = tensorlib.sqrt(qmu_v)

        asimov_mu = 0.0
        asimov_data = generate_asimov_data(
            asimov_mu, self.data, self.pdf, self.init_pars, self.par_bounds
        )
        qmuA_v = qmu(poi_test, asimov_data, self.pdf, self.init_pars, self.par_bounds)
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
    """
    The empirical distribution of the test statistic.

    Unlike ~pyhf.infer.calculators.AsymptoticTestStatDistribution where the
    distribution for the test statistic is normally distributed, the
    :math:`p`-values etc are computed from the sampled distribution.
    """

    def __init__(self, samples):
        """
        Empirical distribution.

        Args:
            samples (Tensor): The test statistics sampled from the distribution.

        Returns:
            ~pyhf.infer.calculators.EmpiricalDistribution: The empirical distribution of test statistic.

        """
        self.samples = samples.ravel()

    def pvalue(self, value):
        """
        Compute the :math:`p`-value for a given value of the test statistic.

        Args:
            value (`float`): The test statistic value.

        Returns:
            Float: The integrated probability to observe a value at least as large as the observed one.

        """
        tensorlib, _ = get_backend()
        return (
            tensorlib.where(self.samples >= value, 1, 0).sum()
            / tensorlib.shape(self.samples)[0]
        )

    def expected_value(self, nsigma):
        """
        Return the expected value of the test statistic.

        Args:
            nsigma (`int` or `tensor`): The number of standard deviations.

        Returns:
            Float: The expected value of the test statistic.
        """
        tensorlib, _ = get_backend()
        import numpy as np

        # TODO: tensorlib.percentile function
        return np.percentile(self.samples, (tensorlib.normal_cdf(nsigma)) * 100)


class ToyCalculator(object):
    """The Toy-based Calculator."""

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
        """
        Toy-based Calculator.

        Args:
            data (`tensor`): The observed data.
            pdf (~pyhf.pdf.Model): The statistical model adhering to the schema ``model.json``.
            init_pars (`tensor`): The initial parameter values to be used for fitting.
            par_bounds (`tensor`): The parameter value bounds to be used for fitting.
            qtilde (`bool`): Whether to use qtilde as the test statistic.
            ntoys (`int`): Number of toys to use (how many times to sample the underlying distributions)
            track_progress (`bool`): Whether to display the `tqdm` progress bar or not (outputs to `stderr`)

        Returns:
            ~pyhf.infer.calculators.ToyCalculator: The calculator for toy-based quantities.

        """
        self.ntoys = ntoys
        self.data = data
        self.pdf = pdf
        self.init_pars = init_pars or pdf.config.suggested_init()
        self.par_bounds = par_bounds or pdf.config.suggested_bounds()
        self.track_progress = track_progress

    def distributions(self, poi_test, track_progress=None):
        """
        Probability Distributions of the test statistic value under the signal + background and background-only hypothesis.

        Args:
            poi_test: The value for the parameter of interest.
            track_progress (`bool`): Whether to display the `tqdm` progress bar or not (outputs to `stderr`)

        Returns:
            Tuple (~pyhf.infer.calculators.EmpiricalDistribution): The distributions under the hypotheses.

        """
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
        """
        Compute the test statistic for the observed data under the studied model.

        Args:
            poi_test: The value for the parameter of interest.

        Returns:
            Float: the value of the test statistic.

        """
        qmu_v = qmu(poi_test, self.data, self.pdf, self.init_pars, self.par_bounds)
        return qmu_v

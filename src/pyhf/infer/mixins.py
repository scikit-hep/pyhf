"""
Mixins for Hypothesis Testing.
"""
from .. import get_backend
from .test_statistics import qmu
from .calculators import AsymptoticTestStatDistribution, EmpiricalDistribution


class Calculator(object):
    def __init__(
        self, data, pdf, init_pars=None, par_bounds=None, qtilde=False, ntoys=2000
    ):
        self.data = data
        self.pdf = pdf
        self.init_pars = init_pars or pdf.config.suggested_init()
        self.par_bounds = par_bounds or pdf.config.suggested_bounds()
        self.qtilde = qtilde
        self.distribution = None

        # TODO: better names???
        # for Asymptotics, it is self.sqrtqmuA_v
        # for Toys, it is signal/bkg qtilde
        self.something_signal = None
        self.something_bkg = None

        # toys
        self.ntoys = ntoys

    def distributions(self, poi_test):
        if self.something_signal is None or self.something_bkg is None:
            raise RuntimeError('need to call .teststatistic(poi_test) first')

        if self.distribution is None:
            raise RuntimeError('need to call this from a mixin\'d class')

        s_plus_b = self.distribution(signal_qtilde)
        b_only = self.distribution(bkg_qtilde)
        return s_plus_b, b_only

    def qmu(self, mu, data=None, pdf=None, init_pars=None, par_bounds=None):
        return qmu(
            mu,
            data or self.data,
            pdf or self.pdf,
            init_pars or self.init_pars,
            par_bounds or self.par_bounds,
        )


class AsymptoticCalculator(Calculator):
    def __init__(self, *args, **kwargs):
        super(AsymptoticCalculator, self).__init__(*args, **kwargs)
        self.distribution = AsymptoticTestStatDistribution

    def teststatistic(self, poi_test):
        tensorlib, _ = get_backend()
        sqrtqmu_v = tensorlib.sqrt(self.qmu(poi_test))

        asimov_mu = 0.0
        asimov_data = generate_asimov_data(
            asimov_mu, self.data, self.pdf, self.init_pars, self.par_bounds
        )
        qmuA_v = self.qmu(poi_test, data=asimov_data)
        self.something_signal = -tensorlib.sqrt(qmuA_v)
        self.something_bkg = 0.0

        if not self.qtilde:  # qmu
            teststat = sqrtqmu_v + self.something_signal
        else:  # qtilde

            def _true_case():
                teststat = sqrtqmu_v + self.something_signal
                return teststat

            def _false_case():
                qmu = tensorlib.power(sqrtqmu_v, 2)
                qmu_A = tensorlib.power(self.something_signal, 2)
                teststat = (qmu_A - qmu) / (2 * self.something_signal)
                return teststat

            teststat = tensorlib.conditional(
                (sqrtqmu_v < self.something_signal), _true_case, _false_case
            )
        return teststat


class ToyCalculator(Calculator):
    def __init__(self, *args, **kwargs):
        super(AsymptoticCalculator, self).__init__(*args, **kwargs)
        self.distribution = EmpiricalDistribution

    def teststatistic(self, poi_test):
        tensorlib, _ = get_backend()
        sample_shape = (self.ntoys,)

        signal_pars = [*self.init_pars]
        signal_pars[self.pdf.config.poi_index] = poi_test
        signal_pdf = self.pdf.make_pdf(tensorlib.astensor(signal_pars))
        signal_sample = signal_pdf.sample(sample_shape)

        bkg_pars = [*self.init_pars]
        bkg_pars[self.pdf.config.poi_index] = 0.0
        bkg_pdf = self.pdf.make_pdf(tensorlib.astensor(bkg_pars))
        bkg_sample = bkg_pdf.sample(sample_shape)

        self.qtilde_signal = tensorlib.astensor(
            self.qmu(poi_test, data=sample, init_pars=signal_pars)
            for sample in signal_sample
        )
        self.qtilde_bkg = tensorlib.astensor(
            self.qmu(poi_test, data=sample, init_pars=bkg_pars) for sample in bkg_sample
        )

        qmu_v = self.qmu(poi_test)

        return qmu_v

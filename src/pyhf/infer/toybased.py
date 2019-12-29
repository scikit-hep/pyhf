from .. import get_backend
from .test_statistics import qmu


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
        self, data, pdf, init_pars=None, par_bounds=None, qtilde=False, ntoys=1000
    ):
        self.ntoys = ntoys
        self.data = data
        self.pdf = pdf
        self.init_pars = init_pars or pdf.config.suggested_init()
        self.par_bounds = par_bounds or pdf.config.suggested_bounds()

    def distributions(self, poi_test):
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

        signal_qtilde = tensorlib.astensor(
            [
                qmu(poi_test, sample, self.pdf, signal_pars, self.par_bounds)
                for sample in signal_sample
            ]
        )
        bkg_qtilde = tensorlib.astensor(
            [
                qmu(poi_test, sample, self.pdf, bkg_pars, self.par_bounds)
                for sample in bkg_sample
            ]
        )
        s_plus_b = EmpiricalDistribution(signal_qtilde)
        b_only = EmpiricalDistribution(bkg_qtilde)
        return s_plus_b, b_only

    def teststatistic(self, poi_test):
        qmu_v = qmu(poi_test, self.data, self.pdf, self.init_pars, self.par_bounds)
        return qmu_v

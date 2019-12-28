from .. import get_backend
from .mle import fixed_poi_fit
from .test_statistics import qmu


def generate_asimov_data(asimov_mu, data, pdf, init_pars, par_bounds):
    """Compute Asimov Dataset (expected yields at best-fit values) for a given POI value."""
    bestfit_nuisance_asimov = fixed_poi_fit(asimov_mu, data, pdf, init_pars, par_bounds)
    return pdf.expected_data(bestfit_nuisance_asimov)


def distributions_from_asymptocics(sqrtqmuA_v):
    splusb = AsymptoticTestStatDistribution(shift=-sqrtqmuA_v)
    bonly = AsymptoticTestStatDistribution(shift=0.0)
    return splusb, bonly


class AsymptoticTestStatDistribution(object):
    def __init__(self, shift):
        self.shift = shift

    def pvalue(self, value):
        tensorlib, _ = get_backend()
        return 1 - tensorlib.normal_cdf(value - self.shift)

    def expected_value(self, nsigma):
        return nsigma


class AsymptoticCalculator(object):
    def __init__(self, data, pdf, init_pars=None, par_bounds=None, qtilde=False):
        tensorlib, _ = get_backend()
        self.data = data
        self.pdf = pdf
        self.init_pars = init_pars or pdf.config.suggested_init()
        self.par_bounds = par_bounds or pdf.config.suggested_bounds()
        self.qtilde = qtilde

        asimov_mu = 0.0
        self.asimov_data = generate_asimov_data(
            asimov_mu, self.data, self.pdf, self.init_pars, self.par_bounds
        )

    def distributions(self, poi_test):
        tensorlib, _ = get_backend()
        qmuA_v = qmu(
            poi_test, self.asimov_data, self.pdf, self.init_pars, self.par_bounds
        )
        self.sqrtqmuA_v = tensorlib.sqrt(qmuA_v)
        s_plus_b, b_only = distributions_from_asymptocics(self.sqrtqmuA_v)
        return s_plus_b, b_only

    def teststatistic(self, poi_test):
        qmu_v = qmu(poi_test, self.data, self.pdf, self.init_pars, self.par_bounds)
        tensorlib, _ = get_backend()
        sqrtqmu_v = tensorlib.sqrt(qmu_v)
        if self.qtilde:

            def _true():
                return sqrtqmu_v - self.sqrtqmuA_v

            def _false():
                return (
                    tensorlib.power(sqrtqmu_v, 2) - tensorlib.power(self.sqrtqmuA_v, 2)
                ) / (2 * self.sqrtqmuA_v)

            return tensorlib.conditional((sqrtqmu_v < self.sqrtqmuA_v), _true, _false)
        else:
            return sqrtqmu_v - self.sqrtqmuA_v

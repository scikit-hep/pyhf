import scipy.stats
import numpy as np
from .. import get_backend


class AsymptoticTestStatDistribution(object):
    def __init__(self, sqrtqmu_A, mu_prime='mu', qtilde=True):
        self.sqrtqmu_A = sqrtqmu_A
        self.mu_prime = mu_prime
        self.qtilde = qtilde
        if not mu_prime in ['mu', 'zero']:
            raise ValueError('...')

    def expected_value(self, nsigma):
        # NOTE:
        # To compute the expected p-value, one would need to first compute a hypothetical
        # observed test-statistic for a dataset whose best-fit value is mu^ = mu'-n*sigma:
        # $q_n$, and the proceed with the normal p-value computation for whatever test-statistic
        # was used. However, we can make a shortcut by just computing the p-values in mu^/sigma
        # space, where the p-values are Clsb = cdf(x-sqrt(lambda)) and CLb=cdf(x)

        tensorlib, _ = get_backend()
        if self.mu_prime == 'zero':
            return tensorlib.normal_cdf(nsigma)
        elif self.mu_prime == 'mu':
            return tensorlib.normal_cdf(nsigma - self.sqrtqmu_A)

    def pvalue(self, cut):
        tensorlib, _ = get_backend()

        qmu = cut
        sqrtqmu_v = tensorlib.sqrt(cut)

        qmu_A = tensorlib.sqrt(self.sqrtqmu_A)
        sqrtqmuA_v = self.sqrtqmu_A

        qtilde = self.qtilde
        if not qtilde:  # qmu
            nullval = sqrtqmu_v
            altval = -(sqrtqmuA_v - sqrtqmu_v)
        else:  # qtilde
            if sqrtqmu_v < sqrtqmuA_v:
                nullval = sqrtqmu_v
                altval = -(sqrtqmuA_v - sqrtqmu_v)
            else:
                qmu = tensorlib.power(sqrtqmu_v, 2)
                qmu_A = tensorlib.power(sqrtqmuA_v, 2)
                nullval = (qmu + qmu_A) / (2 * sqrtqmuA_v)
                altval = (qmu - qmu_A) / (2 * sqrtqmuA_v)
        val = altval if self.mu_prime == 'zero' else nullval
        return 1 - tensorlib.normal_cdf(val)


class EmpiricalTestStatDistribution(object):
    def __init__(self, samples):
        self.samples = samples

    def expected_value(self, nsigma):
        pct = scipy.stats.norm.cdf(nsigma) * 100
        return np.percentile(self.samples, pct)

    def pvalue(self, cut):
        return len(self.samples[self.samples > cut]) / len(self.samples)

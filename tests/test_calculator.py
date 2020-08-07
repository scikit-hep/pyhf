import pyhf
import pyhf.infer.calculators


def test_calc_dist():
    asymptotic_dist = pyhf.infer.calculators.AsymptoticTestStatDistribution(0.0)
    assert asymptotic_dist.pvalue(-1) == 1 - asymptotic_dist.cdf(-1)

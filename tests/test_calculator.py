import pyhf
import pyhf.infer.calculators

def test_calc_distr():
    a = pyhf.infer.calculators.AsymptoticTestStatDistribution(0.0)    
    assert a.pvalue(1) == a.cdf(-1)
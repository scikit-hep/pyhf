import pyhf
import pyhf.infer.calculators
import pytest
import logging


def test_calc_dist():
    asymptotic_dist = pyhf.infer.calculators.AsymptoticTestStatDistribution(0.0)
    assert asymptotic_dist.pvalue(-1) == 1 - asymptotic_dist.cdf(-1)


@pytest.mark.parametrize("calculator", ["asymptotics", "toybased"])
@pytest.mark.parametrize("qtilde", [True, False])
def test_deprecated_qtilde(caplog, mocker, calculator, qtilde):
    with caplog.at_level(logging.WARNING):
        pyhf.infer.utils.create_calculator(
            calculator, ['fake data'], mocker.Mock(), qtilde=qtilde
        )
        assert "is deprecated. Use test_stat" in caplog.text

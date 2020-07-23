import pyhf
import pytest
import logging


@pytest.mark.parametrize('optimizer', ['scipy', 'minuit'])
def test_jax_jit(caplog, optimizer):
    pyhf.set_backend(pyhf.tensor.jax_backend(precision='64b'), optimizer)
    pdf = pyhf.simplemodels.hepdata_like([50.0], [100.0], [10.0])
    data = pyhf.tensorlib.astensor([125.0] + pdf.config.auxdata)

    with caplog.at_level(logging.DEBUG, 'pyhf.optimize.opt_jax'):
        pyhf.infer.mle.fixed_poi_fit(
            1.0, data, pdf, do_grad=True, do_stitch=True, return_fitted_val=True
        )  # jit
        assert 'jitting function' in caplog.text
        caplog.clear()

    with caplog.at_level(logging.DEBUG, 'pyhf.optimize.opt_jax'):
        pyhf.infer.mle.fixed_poi_fit(
            2.0, data, pdf, do_grad=True, do_stitch=True, return_fitted_val=True
        )  # jit
        assert 'jitting function' not in caplog.text

    with caplog.at_level(logging.DEBUG, 'pyhf.optimize.opt_jax'):
        pyhf.infer.mle.fit(
            data, pdf, do_grad=True, do_stitch=True, return_fitted_val=True
        )  # jit
        assert 'jitting function' in caplog.text
        caplog.clear()

    with caplog.at_level(logging.DEBUG, 'pyhf.optimize.opt_jax'):
        pyhf.infer.mle.fit(
            data, pdf, do_grad=True, do_stitch=True, return_fitted_val=True
        )  # jit
        assert 'jitting function' not in caplog.text

    with caplog.at_level(logging.DEBUG, 'pyhf.optimize.opt_jax'):
        pyhf.infer.mle.fixed_poi_fit(
            3.0, data, pdf, do_grad=True, do_stitch=True, return_fitted_val=True
        )  # jit
        assert 'jitting function' not in caplog.text


def test_jax_jit_switch_optimizer(caplog):
    pyhf.set_backend(pyhf.tensor.jax_backend(precision='64b'), 'scipy')
    pdf = pyhf.simplemodels.hepdata_like([50.0], [100.0], [10.0])
    data = pyhf.tensorlib.astensor([125.0] + pdf.config.auxdata)

    with caplog.at_level(logging.DEBUG, 'pyhf.optimize.opt_jax'):
        pyhf.infer.mle.fixed_poi_fit(
            1.0, data, pdf, do_grad=True, do_stitch=True, return_fitted_val=True
        )  # jit
        assert 'jitting function' in caplog.text
        caplog.clear()

    pyhf.set_backend(pyhf.tensorlib, 'minuit')
    with caplog.at_level(logging.DEBUG, 'pyhf.optimize.opt_jax'):
        pyhf.infer.mle.fixed_poi_fit(
            2.0, data, pdf, do_grad=True, do_stitch=True, return_fitted_val=True
        )  # jit
        assert 'jitting function' not in caplog.text

    with caplog.at_level(logging.DEBUG, 'pyhf.optimize.opt_jax'):
        pyhf.infer.mle.fit(
            data, pdf, do_grad=True, do_stitch=True, return_fitted_val=True
        )  # jit
        assert 'jitting function' in caplog.text
        caplog.clear()

    pyhf.set_backend(pyhf.tensorlib, 'scipy')
    with caplog.at_level(logging.DEBUG, 'pyhf.optimize.opt_jax'):
        pyhf.infer.mle.fit(
            data, pdf, do_grad=True, do_stitch=True, return_fitted_val=True
        )  # jit
        assert 'jitting function' not in caplog.text

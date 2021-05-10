import pyhf
import pytest
import logging


@pytest.mark.parametrize(
    'return_fitted_val', [False, True], ids=['no_fitval', 'do_fitval']
)
@pytest.mark.parametrize('do_stitch', [False, True], ids=['no_stitch', 'do_stitch'])
@pytest.mark.parametrize('do_grad', [False, True], ids=['no_grad', 'do_grad'])
@pytest.mark.parametrize('optimizer', ['scipy', 'minuit'])
def test_jax_jit(caplog, optimizer, do_grad, do_stitch, return_fitted_val):
    pyhf.set_backend("jax", optimizer, precision="64b")
    pdf = pyhf.simplemodels.uncorrelated_background([50.0], [100.0], [10.0])
    data = pyhf.tensorlib.astensor([125.0] + pdf.config.auxdata)

    with caplog.at_level(logging.DEBUG, 'pyhf.optimize.opt_jax'):
        pyhf.infer.mle.fixed_poi_fit(
            1.0,
            data,
            pdf,
            do_grad=do_grad,
            do_stitch=do_stitch,
            return_fitted_val=return_fitted_val,
        )  # jit
        assert 'jitting function' in caplog.text
        caplog.clear()

    with caplog.at_level(logging.DEBUG, 'pyhf.optimize.opt_jax'):
        pyhf.infer.mle.fixed_poi_fit(
            2.0,
            data,
            pdf,
            do_grad=do_grad,
            do_stitch=do_stitch,
            return_fitted_val=return_fitted_val,
        )  # jit
        assert 'jitting function' not in caplog.text

    with caplog.at_level(logging.DEBUG, 'pyhf.optimize.opt_jax'):
        pyhf.infer.mle.fit(
            data,
            pdf,
            do_grad=do_grad,
            do_stitch=do_stitch,
            return_fitted_val=return_fitted_val,
        )  # jit
        assert 'jitting function' in caplog.text
        caplog.clear()

    with caplog.at_level(logging.DEBUG, 'pyhf.optimize.opt_jax'):
        pyhf.infer.mle.fit(
            data,
            pdf,
            do_grad=do_grad,
            do_stitch=do_stitch,
            return_fitted_val=return_fitted_val,
        )  # jit
        assert 'jitting function' not in caplog.text

    with caplog.at_level(logging.DEBUG, 'pyhf.optimize.opt_jax'):
        pyhf.infer.mle.fixed_poi_fit(
            3.0,
            data,
            pdf,
            do_grad=do_grad,
            do_stitch=do_stitch,
            return_fitted_val=return_fitted_val,
        )  # jit
        assert 'jitting function' not in caplog.text


@pytest.mark.parametrize(
    'return_fitted_val', [False, True], ids=['no_fitval', 'do_fitval']
)
@pytest.mark.parametrize('do_stitch', [False, True], ids=['no_stitch', 'do_stitch'])
@pytest.mark.parametrize('do_grad', [False, True], ids=['no_grad', 'do_grad'])
def test_jax_jit_switch_optimizer(caplog, do_grad, do_stitch, return_fitted_val):
    pyhf.set_backend("jax", "scipy", precision="64b")
    pdf = pyhf.simplemodels.uncorrelated_background([50.0], [100.0], [10.0])
    data = pyhf.tensorlib.astensor([125.0] + pdf.config.auxdata)

    with caplog.at_level(logging.DEBUG, 'pyhf.optimize.opt_jax'):
        pyhf.infer.mle.fixed_poi_fit(
            1.0,
            data,
            pdf,
            do_grad=do_grad,
            do_stitch=do_stitch,
            return_fitted_val=return_fitted_val,
        )  # jit
        assert 'jitting function' in caplog.text
        caplog.clear()

    pyhf.set_backend(pyhf.tensorlib, 'minuit')
    with caplog.at_level(logging.DEBUG, 'pyhf.optimize.opt_jax'):
        pyhf.infer.mle.fixed_poi_fit(
            2.0,
            data,
            pdf,
            do_grad=do_grad,
            do_stitch=do_stitch,
            return_fitted_val=return_fitted_val,
        )  # jit
        assert 'jitting function' not in caplog.text

    with caplog.at_level(logging.DEBUG, 'pyhf.optimize.opt_jax'):
        pyhf.infer.mle.fit(
            data,
            pdf,
            do_grad=do_grad,
            do_stitch=do_stitch,
            return_fitted_val=return_fitted_val,
        )  # jit
        assert 'jitting function' in caplog.text
        caplog.clear()

    pyhf.set_backend(pyhf.tensorlib, 'scipy')
    with caplog.at_level(logging.DEBUG, 'pyhf.optimize.opt_jax'):
        pyhf.infer.mle.fit(
            data,
            pdf,
            do_grad=do_grad,
            do_stitch=do_stitch,
            return_fitted_val=return_fitted_val,
        )  # jit
        assert 'jitting function' not in caplog.text


@pytest.mark.parametrize(
    'return_fitted_val', [False, True], ids=['no_fitval', 'do_fitval']
)
@pytest.mark.parametrize('do_grad', [False, True], ids=['no_grad', 'do_grad'])
def test_jax_jit_enable_stitching(caplog, do_grad, return_fitted_val):
    pyhf.set_backend("jax", "scipy", precision="64b")
    pdf = pyhf.simplemodels.uncorrelated_background([50.0], [100.0], [10.0])
    data = pyhf.tensorlib.astensor([125.0] + pdf.config.auxdata)

    with caplog.at_level(logging.DEBUG, 'pyhf.optimize.opt_jax'):
        pyhf.infer.mle.fixed_poi_fit(
            1.0,
            data,
            pdf,
            do_grad=do_grad,
            do_stitch=False,
            return_fitted_val=return_fitted_val,
        )  # jit
        assert 'jitting function' in caplog.text
        caplog.clear()

    with caplog.at_level(logging.DEBUG, 'pyhf.optimize.opt_jax'):
        pyhf.infer.mle.fixed_poi_fit(
            1.0,
            data,
            pdf,
            do_grad=do_grad,
            do_stitch=True,
            return_fitted_val=return_fitted_val,
        )  # jit
        assert 'jitting function' in caplog.text
        caplog.clear()


@pytest.mark.parametrize(
    'return_fitted_val', [False, True], ids=['no_fitval', 'do_fitval']
)
@pytest.mark.parametrize('do_stitch', [False, True], ids=['no_stitch', 'do_stitch'])
def test_jax_jit_enable_autograd(caplog, do_stitch, return_fitted_val):
    pyhf.set_backend("jax", "scipy", precision="64b")
    pdf = pyhf.simplemodels.uncorrelated_background([50.0], [100.0], [10.0])
    data = pyhf.tensorlib.astensor([125.0] + pdf.config.auxdata)

    with caplog.at_level(logging.DEBUG, 'pyhf.optimize.opt_jax'):
        pyhf.infer.mle.fixed_poi_fit(
            1.0,
            data,
            pdf,
            do_grad=False,
            do_stitch=do_stitch,
            return_fitted_val=return_fitted_val,
        )  # jit
        assert 'jitting function' in caplog.text
        caplog.clear()

    with caplog.at_level(logging.DEBUG, 'pyhf.optimize.opt_jax'):
        pyhf.infer.mle.fixed_poi_fit(
            1.0,
            data,
            pdf,
            do_grad=True,
            do_stitch=do_stitch,
            return_fitted_val=return_fitted_val,
        )  # jit
        assert 'jitting function' in caplog.text
        caplog.clear()

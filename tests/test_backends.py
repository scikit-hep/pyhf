import pyhf
import jax
import pytest


def test_default_backend():
    pyhf.set_backend("jax", default=True)

    assert pyhf.default_backend.name == 'jax'
    assert pyhf.tensorlib.name == 'jax'


def test_nondefault_backend():
    pyhf.set_backend("jax", default=False)

    assert pyhf.default_backend.name == 'numpy'
    assert pyhf.tensorlib.name == 'jax'


@pytest.mark.parametrize('jitted', (False, True))
def test_diffable_backend(jitted):
    pyhf.set_backend("jax", default=True)

    def example_op(x):
        y = pyhf.default_backend.astensor(x)
        return 2 * y

    if jitted:
        assert jax.jacrev(jax.jit(example_op))([1.0]) == [2.0]
    else:
        assert jax.jacrev(example_op)([1.0]) == [2.0]

    def example_op2(x):
        y = pyhf.default_backend.power(x, 2)
        z = pyhf.tensorlib.sum(y)
        return z

    if jitted:
        assert jax.jacrev(jax.jit(example_op2))(
            pyhf.tensorlib.astensor([2.0, 3.0])
        ).tolist() == [
            4.0,
            6.0,
        ]
    else:
        assert jax.jacrev(example_op2)(
            pyhf.tensorlib.astensor([2.0, 3.0])
        ).tolist() == [
            4.0,
            6.0,
        ]


def test_diffable_backend_failure():
    pyhf.set_backend("numpy", default=True)
    pyhf.set_backend("jax")

    def example_op(x):
        y = pyhf.default_backend.astensor(x)
        return 2 * y

    with pytest.raises(
        (
            ValueError,
            jax._src.errors.TracerArrayConversionError,
            jax.errors.ConcretizationTypeError,
        )
    ):
        jax.jacrev(example_op)([1.0])

    def example_op2(x):
        y = pyhf.default_backend.power(x, 2)
        z = pyhf.tensorlib.sum(y)
        return z

    with pytest.raises(jax._src.errors.TracerArrayConversionError):
        jax.jacrev(example_op2)(pyhf.tensorlib.astensor([2.0, 3.0]))


def test_backend_array_type(backend):
    assert backend[0].array_type is not None


def test_tensor_array_types():
    # can't really assert the content of them so easily
    assert pyhf.tensor.array_types

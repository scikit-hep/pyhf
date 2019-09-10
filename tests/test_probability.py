from pyhf import probability
from pyhf import get_backend


def test_poisson(backend):
    result = probability.Poisson([10.0]).log_prob(2.0)
    assert result.shape == (1,)

    result = probability.Poisson([10.0, 10.0]).log_prob(2.0)
    assert result.shape == (2,)

    result = probability.Poisson([10.0, 10.0]).log_prob([2.0, 3.0])
    assert result.shape == (2,)

    result = probability.Poisson([10.0, 10.0]).log_prob([[2.0, 3.0]])
    assert result.shape == (1, 2)


def test_normal(backend):
    result = probability.Normal([10.0], [1]).log_prob(2.0)
    assert result.shape == (1,)

    result = probability.Normal([10.0, 10.0], [1, 1]).log_prob(2.0)
    assert result.shape == (2,)

    result = probability.Normal([10.0, 10.0], [10.0, 10.0]).log_prob([2.0, 3.0])
    assert result.shape == (2,)

    result = probability.Normal([10.0, 10.0], [10.0, 10.0]).log_prob([[2.0, 3.0]])
    assert result.shape == (1, 2)


def test_joint(backend):
    tensorlib, _ = backend
    p1 = probability.Poisson([10.0]).log_prob(2.0)
    p2 = probability.Poisson([10.0]).log_prob(3.0)
    assert tensorlib.tolist(probability.joint_logpdf([p1, p2])) == tensorlib.tolist(
        p1 + p2
    )


def test_normal(backend):
    tensorlib, _ = backend
    result = probability.Independent(probability.Poisson([10.0, 10])).log_prob(
        [2.0, 3.0]
    )

    p1 = probability.Poisson([10.0]).log_prob(2.0)
    p2 = probability.Poisson([10.0]).log_prob(3.0)
    assert tensorlib.tolist(probability.joint_logpdf([p1, p2]))[0] == tensorlib.tolist(
        result
    )

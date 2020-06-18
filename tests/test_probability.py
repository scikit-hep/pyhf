from pyhf import probability
import numpy as np


def test_poisson(backend):
    tb, _ = backend
    result = probability.Poisson(tb.astensor([10.0])).log_prob(tb.astensor(2.0))
    assert result.shape == (1,)

    result = probability.Poisson(tb.astensor([10.0, 10.0])).log_prob(tb.astensor(2.0))
    assert result.shape == (2,)

    result = probability.Poisson(tb.astensor([10.0, 10.0])).log_prob(
        tb.astensor([2.0, 3.0])
    )
    assert result.shape == (2,)

    result = probability.Poisson(tb.astensor([10.0, 10.0])).log_prob(
        tb.astensor([[2.0, 3.0]])
    )
    assert result.shape == (1, 2)


def test_normal(backend):
    tb, _ = backend
    result = probability.Normal(tb.astensor([10.0]), tb.astensor([1])).log_prob(
        tb.astensor(2.0)
    )
    assert result.shape == (1,)

    result = probability.Normal(
        tb.astensor([10.0, 10.0]), tb.astensor([1, 1])
    ).log_prob(tb.astensor(2.0))
    assert result.shape == (2,)

    result = probability.Normal(
        tb.astensor([10.0, 10.0]), tb.astensor([10.0, 10.0])
    ).log_prob(tb.astensor([2.0, 3.0]))
    assert result.shape == (2,)

    result = probability.Normal(
        tb.astensor([10.0, 10.0]), tb.astensor([10.0, 10.0])
    ).log_prob(tb.astensor([[2.0, 3.0]]))
    assert result.shape == (1, 2)


def test_joint(backend):
    tb, _ = backend
    p1 = probability.Poisson(tb.astensor([10.0])).log_prob(tb.astensor(2.0))
    p2 = probability.Poisson(tb.astensor([10.0])).log_prob(tb.astensor(3.0))
    assert tb.tolist(probability.Simultaneous._joint_logpdf([p1, p2])) == tb.tolist(
        p1 + p2
    )


def test_independent(backend):
    tb, _ = backend
    result = probability.Independent(
        probability.Poisson(tb.astensor([10.0, 10.0]))
    ).log_prob(tb.astensor([2.0, 3.0]))

    p1 = probability.Poisson(tb.astensor([10.0])).log_prob(tb.astensor(2.0))
    p2 = probability.Poisson(tb.astensor([10.0])).log_prob(tb.astensor(3.0))
    assert tb.tolist(probability.Simultaneous._joint_logpdf([p1, p2]))[0] == tb.tolist(
        result
    )
    assert tb.tolist(probability.Simultaneous._joint_logpdf([p1, p2]))[0] == tb.tolist(
        result
    )


def test_simultaneous_list_ducktype():
    myobjs = np.random.randint(100, size=10).tolist()
    sim = probability.Simultaneous(myobjs, None)
    assert sim[3] == myobjs[3]
    for simobj, myobj in zip(sim, myobjs):
        assert simobj == myobj

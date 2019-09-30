from pyhf import probability


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
    tensorlib, _ = backend
    result = probability.Independent(
        probability.Poisson(tensorlib.astensor([10.0, 10.0]))
    ).log_prob(tensorlib.astensor([2.0, 3.0]))

    p1 = probability.Poisson(tensorlib.astensor([10.0])).log_prob(
        tensorlib.astensor(2.0)
    )
    p2 = probability.Poisson(tensorlib.astensor([10.0])).log_prob(
        tensorlib.astensor(3.0)
    )
    assert tensorlib.tolist(probability.Simultaneous._joint_logpdf([p1, p2]))[
        0
    ] == tensorlib.tolist(result)

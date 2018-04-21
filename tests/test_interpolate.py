import pyhf
import numpy as np
import pytest

def test_interpcode_0():
    f = lambda x: pyhf.interpolate.interpolator(0)(at_minus_one = 0.5, at_zero =1, at_plus_one = 2.0, alphas = x)
    assert 1+f(-2) == 0.0
    assert 1+f(-1) == 0.5
    assert 1+f(0) == 1.0
    assert 1+f(1) == 2.0
    assert 1+f(2) == 3.0 #extrapolation

    #broadcasting
    assert [1 + x for x in f([-2,-1,0,1,2]).reshape(-1)] == [0,0.5,1.0,2.0,3.0]

def test_interpcode_1():
    f = lambda x: pyhf.interpolate.interpolator(1)(at_minus_one = 0.9, at_zero =1, at_plus_one = 1.1, alphas = x)
    assert f(-2) == 0.9**2
    assert f(-1) == 0.9
    assert f(0) == 1.0
    assert f(1) == 1.1
    assert f(2) == 1.1**2

    #broadcasting
    assert np.all(f([-2,-1,0,1,2]).reshape(-1) == [0.9**2, 0.9, 1.0, 1.1, 1.1**2])

def test_invalid_interpcode():

    with pytest.raises(pyhf.exceptions.InvalidInterpCode):
        pyhf.interpolate.interpolator('fake')

    with pytest.raises(pyhf.exceptions.InvalidInterpCode):
        pyhf.interpolate.interpolator(1.2)

    with pytest.raises(pyhf.exceptions.InvalidInterpCode):
        pyhf.interpolate.interpolator(-1)

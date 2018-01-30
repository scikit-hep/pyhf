import pyhf
import numpy as np
import json

def test_interpcode_0():
    f = pyhf._hfinterp_code0(at_minus_one = 0.5, at_zero =1, at_plus_one = 2.0)
    assert 1+f(-2) == 0.0
    assert 1+f(-1) == 0.5
    assert 1+f(0) == 1.0
    assert 1+f(1) == 2.0
    assert 1+f(2) == 3.0 #extrapolation

    #broadcasting
    assert [1 + x for x in f([-2,-1,0,1,2])] == [0,0.5,1.0,2.0,3.0]

def test_interpcode_1():
    f = pyhf._hfinterp_code1(at_minus_one = 0.9, at_zero =1, at_plus_one = 1.1)

    assert f(-2) == 0.9**2
    assert f(-1) == 0.9
    assert f(0) == 1.0
    assert f(1) == 1.1
    assert f(2) == 1.1**2

    assert np.all(f([-2,-1,0,1,2]) == [0.9**2, 0.9, 1.0, 1.1, 1.1**2])

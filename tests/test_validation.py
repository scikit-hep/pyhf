import pyhf
import numpy as np
import json

def test_validation_1bin_example_1():
    expected_result = {
        'obs': 0.450337178157,
        'exp': [
            0.06154653039922158,
            0.1472337570386738,
            0.3227412178815565,
            0.5995781547454528,
            0.8636787737204704
        ]
    }
    tolerance = 0.0001


    source = json.load(open('validation/data/1bin_example1.json'))
    pdf  = pyhf.hfpdf.hepdata_like(source['bindata']['sig'], source['bindata']['bkg'], source['bindata']['bkgerr'])
    data = source['bindata']['data'] + pdf.auxdata
    muTest = 1.0
    init_pars  = [1.0]*3 #mu + gam1 + gam2
    par_bounds = [[0,10]] * 3
    clsobs, cls_exp = pyhf.runOnePoint(muTest, data,pdf,init_pars,par_bounds)[-2:]
    cls_obs = 1./clsobs
    cls_exp = [1./x for x in cls_exp]
    assert (cls_obs - expected_result['obs'])/expected_result['obs'] < tolerance
    for result,expected_result in zip(cls_exp, expected_result['exp']):
        assert (result-expected_result)/expected_result < tolerance

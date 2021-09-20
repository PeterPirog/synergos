import numpy as np
from synergos.functions import yeo_johnson_inverse
import pytest
import math


# Make test for lmbda = 0, positive and negative x_inverted
@pytest.mark.parametrize('x_inverted, lmbda, x_original, tol',
                         [
                             (-4., 0., -2., 1e-8),
                             (-1.5, 0., -1., 1e-8),
                             (0., 0., 0., 1e-8),
                             (0.69314718, 0., 1., 1e-8),
                             (1.09861229, 0., 2., 1e-8)
                         ])
def test_lmbda_is_0(x_inverted, lmbda, x_original, tol):
    assert math.isclose(yeo_johnson_inverse(x_inverted, lmbda),
                        x_original, abs_tol=tol)


# Make test for lmbda = 2, positive and negative x_inverted
@pytest.mark.parametrize('x_inverted, lmbda, x_original, tol',
                         [
                             (-1.09861229, 2., -2., 1e-8),
                             (-0.69314718, 2., -1., 1e-8),
                             (0., 2., 0., 1e-8),
                             (1.5, 2., 1., 1e-8),
                             (4., 2., 2., 1e-8)
                         ])
def test_lmbda_is_2(x_inverted, lmbda, x_original, tol):
    assert math.isclose(yeo_johnson_inverse(x_inverted, lmbda),
                        x_original, abs_tol=tol)


# Make test for lmbda = -2, positive and negative x_inverted
@pytest.mark.parametrize('x_inverted, lmbda, x_original, tol',
                         [
                             (-20, -2., -2., 1e-8),
                             (- 3.75, -2., -1., 1e-8),
                             (0., -2., 0., 1e-8),
                             (0.375, -2., 1., 1e-8),
                             (0.44444444, -2., 2., 1e-6)
                         ])
def test_lmbda_is_minus2(x_inverted, lmbda, x_original, tol):
    assert math.isclose(yeo_johnson_inverse(x_inverted, lmbda),
                        x_original, abs_tol=tol)


# Make test for lmbda = 3, positive and negative x_inverted
@pytest.mark.parametrize('x_inverted, lmbda, x_original, tol',
                         [
                             (-0.66666667, 3., -2., 1e-7),
                             (-0.5, 3., -1., 1e-8),
                             (0., 3., 0., 1e-8),
                             (2.33333333, 3., 1., 1e-8),
                             (8.66666667, 3., 2., 1e-8)
                         ])
def test_lmbda_is_3(x_inverted, lmbda, x_original, tol):
    assert math.isclose(yeo_johnson_inverse(x_inverted, lmbda),
                        x_original, abs_tol=tol)


# Make test for  iterable x
def test_x_is_iterable():
    x_inverted = [-20, 0]
    lmbda = -2.
    x_original = [-2, 0]
    assert sum(yeo_johnson_inverse(x_inverted, lmbda)
               - np.array(x_original)) == 0

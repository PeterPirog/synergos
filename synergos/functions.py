import numpy as np
from collections.abc import Iterable


def yeo_johnson_inverse(x, lmbda):
    """Return inverse-transformed input x following Yeo-Johnson inverse
    transform with parameter lambda.

    Example:
        from synergos.functions import yeo_johnson_inverse

        x=[-0.66666667, -0.5, 0., 2.33333333, 8.66666667]
        lmbda=2.5
        y=yeo_johnson_inverse(x,lmbda)
        print(y)
        output >> [-1.25000001 -0.77777778  0.          1.15701439  2.4846171 ]
    """

    if isinstance(lmbda, Iterable):
        out = "lmbda can't be iterable"
        print(out)
        return None

    if not isinstance(x, Iterable):
        x = [x]

    out = np.zeros_like(x)

    for idx, xi in enumerate(x):
        try:
            if xi >= 0 and lmbda == 0:
                out[idx] = np.exp(xi) - 1
            elif xi >= 0 and lmbda != 0:
                out[idx] = np.power(xi * lmbda + 1, 1 / lmbda) - 1
            elif xi < 0 and lmbda == 2:
                out[idx] = 1 - np.exp(-xi)
            elif xi < 0 and lmbda != 2:
                out[idx] = 1 - np.power(-(2 - lmbda) * xi + 1, 1 / (2 - lmbda))
        except ValueError:
            print('Wrong values of arguments.\n'
                  ' lmbda should be float or int, x should be float or int ')
    return out


if __name__ == '__main__':
    x = [-0.66666667, -0.5, 0., 2.33333333, 8.66666667]
    lmbda = 2.5
    y = yeo_johnson_inverse(x, lmbda)
    print(y)

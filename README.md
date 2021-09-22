# synergos
set of tools to improve ML tasks in sklearn and tensorflow

## Installation

Run the following to install:
```bash
$ pip install synergos
```

## Usage of functions
### PercentileTargetEncoder

PercentileTargetEncoder is used for encoding categorical values by changing values to percentiles of distribution
inspration from: https://maxhalford.github.io/blog/target-encoding/

THE MOST IMPORTANT FEATURES OF  THE ENCODER:
1. DO NOT EXPAND NUMBER OF FEATURES IN UNCONTROLLED WAY FOR HIGH CARDINALITY DATA LIKE IN ONE-HOT ENCODING
2. AUTOMATICALLY ENCODE RARE, MISSING AND UNKNOWN LABELS
3. OVERFITTING CONTROLLED BY PARAMETER: m, 1<=m<np.Inf
4. DISTRIBUTION SHAPE CAN BE ANALYSED BY PARAMETER: p  (p=[0.1, 0.5, 0.9])
5. COMPATIBILITY WITH SKLEARN PIPELINES
6. NEEDS 'TARGET' VALUES - FOR USE IN SUPERVISED LEARNING
7. 'TARGET' DISTRIBUTION SHAPE CEN BE NORMALIZED AUTOMATICALLY BY PARAMETER use_internal_yeo_johnson=True

PARAMETERS
features: list of features for encoding, if None transformer find categorical values automatically

ignored_features: list of ignored features when features are found automatically with option features = None

p: percentil value to calculate new feature, default value is 0.5 (median), range  0<p<1, single value or list of values

m: regularization parameter to prevent overfitting, default value is 1, int in range for 1 to np.inf, single value or list of values

remove_original: if True original categorical value is dropped, default value is True

return_df: if True pd.Dataframe as return, if False np.array as return, default value is True

use_internal_yeo_johnson: if True, yeo-johnson transformation is used to normalize 'Target' before encoding, default value is True
```python
import pandas as pd
from synergos.transformers import PercentileTargetEncoder

if __name__ == '__main__':
    df = pd.DataFrame({
        'x_0': ['a'] * 5 + ['b'] * 5,
        'x_1': ['c'] * 9 + ['d'] * 1,
        'y': [1, 1, 1, 1, 0, 1, 0, 0, 0, 0]
    })
    print(df.head(10))
    pte = PercentileTargetEncoder(features=None,
                                  ignored_features=None,
                                  p=[0.8], m=[3],
                                  remove_original=True,
                                  return_df=True,
                                  use_internal_yeo_johnson=False)
    out = pte.fit_transform(X=df[['x_0', 'x_1']], y=df['y'])
    print(out)

    df_test = pd.DataFrame({
        'x_0': ['a'] * 5 + ['V'] * 1 + ['b'] * 4,
        'x_1': ['c'] * 8 + [''] * 1 + ['d'] * 1,
        'y': [1, 1, 1, 1, 0, 1, 0, 0, 0, 0]
    })
    out_test = pte.transform(X=df_test)
    print(out_test)
```

### yeo_johnson_inverse
param: 
features: list of features for encoding, if None transformer find categorical values automatically

```python
from synergos.functions import yeo_johnson_inverse
if __name__ == '__main__':
    x = [-0.66666667, -0.5, 0., 2.33333333, 8.66666667]
    lmbda = 2.5
    y = yeo_johnson_inverse(x, lmbda)
    print(y)
```

# Developing Synergos
To install Synergos, along the tools you need to develop and run tests, run the following in your virtualenv:
```bash
$ pip install -e .[dev]
```

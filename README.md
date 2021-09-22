# synergos
set of tools to improve ML tasks in sklearn and tensorflow

## Installation

Run the following to install:
```bash
$ pip install synergos
```

## Usage of functions
### PercentileTargetEncoder
param: 
    features: list of features for encoding, if None transformer find categorical values automatically
param: 
    ignored_features: list of ignored features when features are found automatically with option features = None
param: 
    p: percentil value to calculate new feature, default value is 0.5 (median), range  0<p<1, single value or list of values
param: 
    m: regularization parameter to prevent overfitting, default value is 1, int in range for 1 to np.inf, single value or list of values
param: 
    remove_original: if True original categorical value is dropped, default value is True
param: 
    return_df: if True pd.Dataframe as return, if False np.array as return, default value is True
param: 
    use_internal_yeo_johnson: if True, yeo-johnson transformation is used to normalize 'Target' before encoding, default value is False
```python
    import pandas as pd
    from synergos.transformers import PercentileTargetEncoder
    df = pd.DataFrame({
        'x_0': ['a'] * 5 + ['b'] * 5,
        'x_1': ['c'] * 9 + ['d'] * 1,
        'y': [1, 1, 1, 1, 0, 1, 0, 0, 0, 0]
    })
    print(df.head())
    pte = PercentileTargetEncoder(features=None,
                                  ignored_features=None,
                                  p=[0.5], m=[1],
                                  remove_original=True,
                                  return_df=True,
                                  use_internal_yeo_johnson=False)
    out = pte.fit_transform(X=df[['x_0', 'x_1']], y=df['y'])
    print(out)
    # dataframe with unknown value 'V' in column x_0
    # and missing value in column x_1
    df_test = pd.DataFrame({
        'x_0': ['a'] * 5 + ['V'] * 1 + ['b'] * 4,
        'x_1': ['c'] * 8 + [''] * 1 + ['d'] * 1,
        'y': [1, 1, 1, 1, 0, 1, 0, 0, 0, 0]
    })
    out_test = pte.transform(X=df_test)
    print(out_test)



```
#Developing Synergos
To install Synergos, along the tools you need to develop and run tests, run the following in your virtualenv:
```bash
$ pip install -e .[dev]
```

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
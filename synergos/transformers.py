import pandas as pd
import numpy as np
from scipy import stats

from sklearn.base import BaseEstimator, TransformerMixin
pd.set_option('display.max_columns', None)


# https://maxhalford.github.io/blog/target-encoding/
class PercentileTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, features=None,
                 ignored_features=None,
                 p=0.5,
                 m=1,
                 remove_original=True,
                 return_df=True,
                 use_internal_yeo_johnson=False,
                 verbose=True):
        super().__init__()
        self.features = features  # selected categorical features
        self.ignored_features = ignored_features
        self.columns = None  # all columns in df
        self.column_target = None
        self.p = p
        self.m = m
        self.N = None  # Number of rows in training dataset
        self.remove_original = remove_original
        self.return_df = return_df
        # usage of yeo-johnson transformation inside encoder
        self.use_internal_yeo_johnson = use_internal_yeo_johnson
        self.verbose = verbose
        # dict with unique values lists for specified feature, key form (feature)
        self.features_unique = {}
        # stored quantiles for whole dataset, key form (p)
        self.global_quantiles = {}
        # stored quantiles for all values, key form (feature, value, p)
        self.value_quantiles = {}
        # stored counts of every value in train data key form (feature, value)
        self.value_counts = {}

        # convert p and m to lists for iteration available
        if isinstance(p, int) or isinstance(p, float):
            self.p = [self.p]
        if isinstance(m, int) or isinstance(m, float):
            self.m = [self.m]

        # convert feature lists for iteration available
        if not isinstance(self.features, list) and self.features is not None:
            self.features = [self.features]

        if not isinstance(self.ignored_features, list) and self.ignored_features is not None:
            self.ignored_features = [self.ignored_features]

    def fit(self, X, y=None):
        X = X.copy()
        # Convert y to proper datatype
        # if y is pd.Series
        if isinstance(y, pd.Series):
            y = y.to_frame().copy()
        # if y is np.array
        elif isinstance(y, type(np.array([0]))):
            y = pd.DataFrame(y, columns=['target']).copy()
        # if y is pd.DataFrame
        elif isinstance(y, pd.DataFrame):
            y = y.copy()
        else:
            print("Wrong target 'y' data type")

        # use yeo-johnson transformation for target inside encoder
        if self.use_internal_yeo_johnson:
            y = stats.yeojohnson(y)[0]  # second component is lambda
            y = pd.DataFrame(y, columns=['target']).copy()

        # Count number of rows in training dataset
        self.N = len(y)

        self.columns = X.columns
        # Find only categorical columns if not defines
        # Auto-search categorical features
        if self.features is None:
            self.features = [col for col in self.columns if X[col].dtypes == 'O']
        else:
            # convert single feature name to list for iteration possibility
            if isinstance(self.features, str):
                self.features = [self.features]

        # Remove ignored features
        if self.ignored_features is not None:
            for ignored_feature in self.ignored_features:
                self.features.remove(ignored_feature)

        if self.verbose and X.isnull().values.any():
            print('There were some nan values if specified features. Nan values are replaced')

        # Replace nan values in selected categorical features by 'MISSING" value
        X[self.features] = X[self.features].fillna('MISSING').copy()

        # Find unique values for specified features
        for feature in self.features:
            self.features_unique[feature] = list(X[feature].unique())
            # add 'UNKNOWN' value for transform never seen values
            self.features_unique[feature].append('UNKNOWN')

            # add 'MISSING' value whole data  were complete and 'MISSING' key is not created
            if not'MISSING' in self.features_unique[feature]:
                self.features_unique[feature].append('MISSING')

        # Find quantiles for all dataset for each value of p
        for p in self.p:
            self.global_quantiles[p] = np.quantile(y, p)

        # Find quantiles for every feature and every value
        for feature in self.features:
            unique_vals_for_feature = self.features_unique[feature]
            # for every unique value for feature
            for value in unique_vals_for_feature:
                value_counts = X.loc[X[feature] == value, feature].count()

                # value not exist in training data
                if value_counts == 0:
                    for p in self.p:
                        # replace missing value by quantile for all data
                        self.value_quantiles[feature, value, p] = self.global_quantiles[p]

                        # set value 1 for 'UNKNOWN'  and 'MISSING'values
                        self.value_counts[feature, value] = 1

                # value exist in training data, quantile can be calculated
                else:
                    # Find y values for specified feature and specified value
                    idx = X[feature] == value
                    # value_not_exist_in_data=sum(idx.astype(int))
                    y_for_value = y[idx].copy()
                    # counts for every feature and every value
                    self.value_counts[feature, value] = len(y_for_value)

                    for p in self.p:
                        # interpolation{‘linear’, ‘lower’, ‘higher’, ‘midpoint’, ‘nearest’}
                        quantile=np.quantile(y_for_value, p,interpolation='linear')
                        print(f'feature={feature}, value={value}, p={p},quantile={quantile}, counts={self.value_counts[feature, value]} \n '
                              f'eta={self.value_counts[feature, value]/self.N}'
                              f'\n y={y_for_value}')
                        self.value_quantiles[feature, value, p] =quantile
        return self

    def transform(self, X):
        X = X.copy()
        # Replace nan values in selected categorical features by 'MISSING" value
        X[self.features] = X[self.features].fillna('MISSING')

        for feature in self.features:
            # Replace never seen values as 'UNKNOWN'
            X[feature] = X[feature].apply(lambda value:
                                          value if value in self.features_unique[feature] else 'UNKNOWN')
            for p in self.p:
                for m in self.m:
                    # Prepare new columns names for percentile values
                    feature_name = feature + '_' + str(p) + '_' + str(m)
                    X[feature_name] = X[feature].apply(lambda value:
                                                       self.__calculate_new_target(feature, value, p, m))

        # Remove original features
        if self.remove_original:
            X = X.drop(self.features, axis=1)

        # Return dataframe or np array
        if self.return_df:
            return X
        else:
            return X.to_numpy()

    def __calculate_new_target(self, feature, value, p, m):
        """
        :param feature: current feature name
        :param value: current value for feature
        :param p: percentile value
        :param m: regularization parameter to prevent overfitting , int in range for 1 to np.inf
        :return: calculated target to replace categorical value
        """
        # N - total number of rows in training set
        N = self.N
        # ni - number of rows with specified 'value' in training set
        ni = self.value_counts[feature, value]
        # eta - proportion of specified value to all values in training set
        eta = ni/N
        # q - quantile value for specified 'feature' and specified 'value'
        q = self.value_quantiles[feature, value, p]
        # mQ - mean quantile for whole dataset
        mQ = self.global_quantiles[p]
        # print(f'eth={eta}')
        return (mQ+m*eta*q)/(1+m*eta)


if __name__ == '__main__':
    df = pd.DataFrame({
        'x_0': ['a'] * 5 + ['b'] * 5,
        'x_1': ['c'] * 9 + ['d'] * 1,
        # 'x_0': ['a','a','a','a','a','b','b','b','b','b'] ,
        # 'x_1': ['c','c','c','c','c','c','c','c','c','d'],
        'y': [1, 1, 1, 1, 0, 1, 0, 0, 0, 0]
    })
    print('aaaaaaa',df.head(10))
    pte = PercentileTargetEncoder(features=None,
                                  ignored_features=None,
                                  p=[0.8], m=[3],
                                  remove_original=True,
                                  return_df=True,
                                  use_internal_yeo_johnson=False)
    out = pte.fit_transform(X=df[['x_0', 'x_1']], y=df['y'])
    print(out)
    print(pte.global_quantiles)
    print(pte.value_quantiles)
    # dataframe with unknown value 'V' in column x_0
    # and missing value in column x_1
    """

    df_test = pd.DataFrame({
        'x_0': ['a'] * 5 + ['V'] * 1 + ['b'] * 4,
        'x_1': ['c'] * 8 + [''] * 1 + ['d'] * 1,
        'y': [1, 1, 1, 1, 0, 1, 0, 0, 0, 0]
    })
    out_test = pte.transform(X=df_test)
    print(out_test)
"""
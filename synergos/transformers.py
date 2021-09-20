import pandas as pd
import numpy as np
from scipy import stats

from sklearn.base import BaseEstimator, TransformerMixin


class CategoriclalQuantileEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, features=None, ignored_features=None, p=0.5, m=1, remove_original=True, return_df=True,
                 handle_missing_or_unknown='return_nan',use_internal_yeo_johnson=False):
        super().__init__()
        self.features = features  # selected categorical features
        self.ignored_features = ignored_features
        self.columns = None  # all columns in df
        self.column_target = None
        self.p = p
        self.m = m
        self.remove_original = remove_original
        self.return_df = return_df
        self.handle_missing_or_unknown = handle_missing_or_unknown  # 'value' or ‘return_nan’
        self.use_internal_yeo_johnson=use_internal_yeo_johnson  # usage of yeo-johnson transformation inside encoder

        self.features_unique = {}  # dict with unique values lists for specified feature, key form (feature)
        self.global_quantiles = {}  # stored quantiles for whole dataset, key form (p)
        self.value_quantiles = {}  # stored quantiles for all values, key form (feature, value, p)
        self.value_counts = {}  # stored counts of every value in train data key form (feature, value)

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
        X=X.copy()
        #if y is pd.Series
        if isinstance(y, pd.Series):
            y = y.to_frame().copy()
        elif isinstance(y, type(np.array([0]))):
            y=pd.DataFrame(y,columns=['target']).copy()
        elif isinstance(y, pd.DataFrame):
            y = y.copy()
        else:
            print("Wrong target 'y' data type")

        # use yeo-johnson transformation for target inside encoder
        if self.use_internal_yeo_johnson:
            y=stats.yeojohnson(y)[0] # second component is lambda
            y = pd.DataFrame(y, columns=['target']).copy()


        self.columns = X.columns
        # Find only categorical columns if not defines
        if self.features is None:
            self.features = [col for col in self.columns if X[col].dtypes == 'O']
            print(f'self.features={self.features}')
        else:
            if isinstance(self.features, str):  # convert single feature name to list for iteration possibility
                self.features = [self.features]

        # Remove ignored features
        if self.ignored_features is not None:
            for ignored_feature in self.ignored_features:
                self.features.remove(ignored_feature)

        #Replace nan values in selected categorical features
        if self.handle_missing_or_unknown == 'return_nan':
            X[self.features] = X[self.features].fillna('MISSING').copy()
        # Concatenate X and y to prevent shifing between rows
        Xy = pd.concat([X, y], axis=1)






        # Find unique values for specified features
        for feature in self.features:
            self.features_unique[feature] = list(X[feature].unique())

        # Find quantiles for all dataset for each value of p
        for p in self.p:
            self.global_quantiles[p] = np.quantile(y, p)




        # Find quantiles for every feature and every value
        for feature in self.features:

            unique_vals_for_feature=list(X[feature].unique())
            print(f'unique_vals_for_feature={unique_vals_for_feature}')

            #SOME ERROR


            for value in unique_vals_for_feature:  # for every unique value for feature
                # Find y values for specified feature and specified value
                idx = Xy[feature] == value
                y_group = y[idx]
                # counts for every feature and every value
                self.value_counts[feature, value] = len(y_group)
                # print(f'n={self.value_counts[feature, value]} for feature={feature}, value={value}')
                print(y_group.isnull().values.any())
                for p in self.p:
                    try:
                        np.quantile(y_group, p)
                    except:
                        print(f' BROKEN FOR feature={feature}, value={value}, counts={len(y_group)}')
                    # print(f'feature={feature}, value={value}, counts={len(y_group)}, quantile={np.quantile(y_group, p)}')
                    self.value_quantiles[feature, value, p] = np.quantile(y_group, p)



        return self

    def transform(self, X):
        X = X.copy()

        # Create new columns for quantile values
        for feature in self.features:
            X[feature] = X[feature].replace(np.nan, 'MISSING')
            X[feature] = X[feature].apply(lambda value: value if value in self.features_unique[feature] else 'UNKNOWN')
            for p in self.p:
                for m in self.m:
                    # Prepare new columns names
                    feature_name = feature + '_' + str(p) + '_' + str(m)

                    # return global quantile values if input value is nan or unknown
                    if self.handle_missing_or_unknown == 'value':
                        X[feature_name] = self.global_quantiles[p]
                        X[feature_name] = X[feature].apply(lambda value: self.global_quantiles[p]
                        if value == "MISSING" or value == 'UNKNOWN'
                        # Quantile Encoder: Tackling High Cardinality Categorical Features in Regression Problems, equation 2
                        else (self.value_counts[feature, value] * self.value_quantiles[feature, value, p] +
                              m * self.global_quantiles[p]) / (self.value_counts[feature, value] + m))

                    # return nan if input value is nan or unknown
                    if self.handle_missing_or_unknown == 'return_nan':
                        X[feature_name] = self.global_quantiles[p]
                        X[feature_name] = X[feature].apply(lambda value: np.nan
                        if value == "MISSING" or value == 'UNKNOWN'
                        # Quantile Encoder: Tackling High Cardinality Categorical Features in Regression Problems, equation 2
                        else (self.value_counts[feature, value] * self.value_quantiles[feature, value, p] +
                              m * self.global_quantiles[p]) / (self.value_counts[feature, value] + m))

        # Remove original features
        if self.remove_original:
            X = X.drop(self.features, axis=1)

        # Return dataframe or np array
        if self.return_df:
            return X
        else:
            return X.to_numpy()

class PercentileTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, features=None, ignored_features=None, p=0.5, m=1, remove_original=True, return_df=True,
                 handle_missing_or_unknown='return_nan',use_internal_yeo_johnson=False,verbose=True):
        super().__init__()
        self.features = features  # selected categorical features
        self.ignored_features = ignored_features
        self.columns = None  # all columns in df
        self.column_target = None
        self.p = p
        self.m = m
        self.N=None # Number of rows in training dataset
        self.remove_original = remove_original
        self.return_df = return_df
        #self.handle_missing_or_unknown = handle_missing_or_unknown  # 'value' or ‘return_nan’
        self.use_internal_yeo_johnson=use_internal_yeo_johnson  # usage of yeo-johnson transformation inside encoder
        self.verbose=verbose

        self.features_unique = {}  # dict with unique values lists for specified feature, key form (feature)
        self.global_quantiles = {}  # stored quantiles for whole dataset, key form (p)
        self.value_quantiles = {}  # stored quantiles for all values, key form (feature, value, p)
        self.value_counts = {}  # stored counts of every value in train data key form (feature, value)

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
        X=X.copy()
        # Convert y to proper datatype
            #if y is pd.Series
        if isinstance(y, pd.Series):
            y = y.to_frame().copy()
            # if y is np.array
        elif isinstance(y, type(np.array([0]))):
            y=pd.DataFrame(y,columns=['target']).copy()
            # if y is pd.DataFrame
        elif isinstance(y, pd.DataFrame):
            y = y.copy()
        else:
            print("Wrong target 'y' data type")

        # use yeo-johnson transformation for target inside encoder
        if self.use_internal_yeo_johnson:
            y=stats.yeojohnson(y)[0] # second component is lambda
            y = pd.DataFrame(y, columns=['target']).copy()

        # Count number of rows in training dataset
        self.N=len(y)


        self.columns = X.columns
        # Find only categorical columns if not defines
            # Auto-search categorical features
        if self.features is None:
            self.features = [col for col in self.columns if X[col].dtypes == 'O']
            # print(f'self.features={self.features}')
        else:
            if isinstance(self.features, str):  # convert single feature name to list for iteration possibility
                self.features = [self.features]

        # Remove ignored features
        if self.ignored_features is not None:
            for ignored_feature in self.ignored_features:
                self.features.remove(ignored_feature)

        if self.verbose and X.isnull().values.any():
            print('There were some nan values if specified features. Nan values are replaced')

        #Replace nan values in selected categorical features by 'MISSING" value
        X[self.features] = X[self.features].fillna('MISSING').copy()

        # Concatenate X and y to prevent shifing between rows
        #Xy = pd.concat([X, y], axis=1)






        # Find unique values for specified features
        for feature in self.features:
            self.features_unique[feature] = list(X[feature].unique())
            # add 'UNKNOWN' value for transform never seen values
            self.features_unique[feature].append('UNKNOWN')

            # add 'MISSING' value whole data  were complete and 'MISSING' key is not created
            if not'MISSING' in self.features_unique[feature]:
                self.features_unique[feature].append('MISSING')

        print(self.features_unique)

        # Find quantiles for all dataset for each value of p
        for p in self.p:
            self.global_quantiles[p] = np.quantile(y, p)


        # Find quantiles for every feature and every value
        for feature in self.features:
            unique_vals_for_feature = self.features_unique[feature]


            # unique_vals_for_feature=list(X[feature].unique())
            #print(f'unique_vals_for_feature={unique_vals_for_feature}')



            for value in unique_vals_for_feature:  # for every unique value for feature

                #Count valu occurencies to detect UNKNOWN  and MISSIN =0
                #value_counts=Xy.loc[Xy[feature] == value, feature].count()
                value_counts = X.loc[X[feature] == value, feature].count()

                # value not exist in training data
                if value_counts == 0:
                    for p in self.p:
                        # replace missing value by quantile for all data
                        self.value_quantiles[feature, value, p] = np.quantile(y, p)
                # value exist in training data, quantile can be calculated
                else:
                    #print(f' Counts feature={feature}, value={value}, counts={value_counts}')

                    # Find y values for specified feature and specified value
                    idx = X[feature] == value

                    value_not_exist_in_data=sum(idx.astype(int))
                    #print(sum(idx.astype(int)))

                    y_group = y[idx]
                    # counts for every feature and every value
                    self.value_counts[feature, value] = len(y_group)
                    # print(f'n={self.value_counts[feature, value]} for feature={feature}, value={value}')
                    # print(y_group.isnull().values.any())
                    for p in self.p:
                        self.value_quantiles[feature, value, p] = np.quantile(y_group, p)

        print('-------------------------------------------')
        print(f'self.value_quantiles={self.value_quantiles}')
        return self

    def transform(self, X):
        X = X.copy()
        #Replace nan values in selected categorical features by 'MISSING" value
        X[self.features] = X[self.features].fillna('MISSING')


        for feature in self.features:
            #X[feature] = X[feature].replace(np.nan, 'MISSING')
            # Replace never seen values as 'UNKNOWN'
            X[feature] = X[feature].apply(lambda value: value if value in self.features_unique[feature] else 'UNKNOWN')
            for p in self.p:
                for m in self.m:
                    # Prepare new columns names for percentile values
                    feature_name = feature + '_' + str(p) + '_' + str(m)

                    mean_quantile=self.global_quantiles[p]
                    # w_quantile=self.value_quantiles[feature, value, p]

                    # return global quantile values if input value is nan or unknown
                    X[feature_name] = X[feature].apply(lambda value: (self.N*mean_quantile+m*self.value_quantiles[feature, value, p])/(self.N+m) )



        # Remove original features
        if self.remove_original:
            X = X.drop(self.features, axis=1)

        # Return dataframe or np array
        if self.return_df:
            return X
        else:
            return X.to_numpy()

pd.set_option('display.max_columns', None)
if __name__ == '__main__':
    df_train = pd.read_csv('train.csv')
    print(df_train.head())

    features=['LotShape','KitchenQual','BsmtFinType1','BsmtUnfSF']
    target=['SalePrice']

    x_train=df_train[features]
    y_train=df_train[target]

    pte=PercentileTargetEncoder(features=None, ignored_features=None, p=[0.5,0.9], m=2, remove_original=True, return_df=True,
                 handle_missing_or_unknown='return_nan', use_internal_yeo_johnson=True)

    # handle_missing_or_unknown='value' or 'return_nan'
    out_train=pte.fit_transform(X=x_train,y=y_train)



    df_test = pd.read_csv('test.csv')
    x_test = df_test[features]
    out_test=pte.transform(X=x_test)
    print(out_test)
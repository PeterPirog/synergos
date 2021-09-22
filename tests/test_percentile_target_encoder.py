import numpy as np
import pandas as pd
from synergos.transformers import PercentileTargetEncoder


def get_calculated_outputs(sheet_name, m, p):
    data = pd.read_excel('categorical_target_encoder_data.xlsx',
                         sheet_name='Data')
    pte = PercentileTargetEncoder(features=None,
                                  ignored_features=None,
                                  p=p,
                                  m=m,
                                  remove_original=True,
                                  return_df=True,
                                  use_internal_yeo_johnson=False,
                                  verbose=True)
    calculated_values = pte.fit_transform(X=data[['x_0', 'x_1']],
                                          y=data['y']).to_numpy()
    correct_values = pd.read_excel('categorical_target_encoder_data.xlsx',
                                   sheet_name=sheet_name).to_numpy()
    return calculated_values, correct_values


def test_encoding_is_set1_correct():
    calculated_values, correct_values = \
        get_calculated_outputs(sheet_name='Set1', m=2, p=0.5)
    differences = np.abs(np.sum(calculated_values - correct_values))
    assert differences <= 1e-8


def test_encoding_is_set2_correct():
    calculated_values, correct_values =\
        get_calculated_outputs(sheet_name='Set2', m=5, p=0.5)
    differences = np.abs(np.sum(calculated_values - correct_values))
    assert differences <= 1e-8

def test_encoding_is_set3_correct():
    calculated_values, correct_values =\
        get_calculated_outputs(sheet_name='Set3', m=3, p=0.8)
    print(f'calculated_values={calculated_values}')
    print(f'correct_values={correct_values}')
    differences = np.abs(np.sum(calculated_values - correct_values))
    assert differences <= 1e-8

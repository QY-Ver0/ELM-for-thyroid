import os

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
import pandas as pd

def get_data(fname='data/thyroid_diff.csv',index_col=None):
    if os.path.isfile(fname):
        df = pd.read_csv(fname,index_col=index_col)
    else:
        raise FileNotFoundError(fname)

    return df

def get_XY(df, test_col='Recurred'):
    if type(df) is pd.DataFrame:
        return df.drop(test_col, axis=1), df[test_col]
    else:
        return df[:,:test_col], df[:,test_col]

def preprocess(df, do_onehot=True, do_scale=False):
    # assume train and test be DataFrames split, without earlier preprocessing
    # Firstly, change the name of this column, authors' grammar mistake
    # train_preprocessed = train.copy()
    # test_preprocessed = test.copy() if test is not None else train.copy()
    df_preprocessed = df.copy()
    # One Hot encoding
    if do_onehot:
        # train_preprocessed.rename(columns={'Hx Radiothreapy': 'Hx Radiotherapy'}, inplace=True)
        # test_preprocessed.rename(columns={'Hx Radiothreapy': 'Hx Radiotherapy'}, inplace=True)
        df_preprocessed.rename(columns={'Hx Radiothreapy': 'Hx Radiotherapy'}, inplace=True)

        cat_risk = [['Low', 'Intermediate', 'High']]
        cat_T = [['T1a', 'T1b', 'T2', 'T3a', 'T3b', 'T4a', 'T4b']]
        cat_N = [['N0', 'N1a', 'N1b']]
        cat_M = [['M0', 'M1']]
        cat_stage = [['I', 'II', 'III', 'IVA', 'IVB']]
        onehot_cols1 = ['Gender', 'Smoking', 'Hx Smoking', 'Hx Radiotherapy', 'Thyroid Function',
                        'Physical Examination',
                        'Adenopathy', 'Pathology', 'Focality']
        onehot_cols2 = ['Response', 'Recurred']

        preprocessor = ColumnTransformer(
            transformers=[
                # ('minmax_age', age_scaler, ['Age']),
                ('age_pass', 'passthrough', ['Age']),
                ('onehot1', OneHotEncoder(categories='auto', drop='if_binary', dtype=bool, sparse_output=False), onehot_cols1),
                ('encode_risk',  Pipeline([
                    ('ord_r', OrdinalEncoder(categories=cat_risk)), ('scale_r', MinMaxScaler()),
                ]),  ['Risk']),
                ('encode_T',     Pipeline([
                    ('ord_T', OrdinalEncoder(categories=cat_T)),    ('scale_r', MinMaxScaler())
                ]),     ['T']),
                ('encode_N',     Pipeline([
                    ('ord_N', OrdinalEncoder(categories=cat_N)),    ('scale_N', MinMaxScaler())
                ]),     ['N']),
                ('encode_M',     Pipeline([
                    ('ord_M', OrdinalEncoder(categories=cat_M)),    ('scale_M', MinMaxScaler())
                ]),     ['M']),
                ('encode_stage', Pipeline([
                    ('ord_s', OrdinalEncoder(categories=cat_stage)),('scale_s', MinMaxScaler())
                ]), ['Stage']),
                ('onehot2', OneHotEncoder(categories='auto', drop='if_binary', dtype=bool, sparse_output=False), onehot_cols2),
            ],
            remainder='passthrough',
            force_int_remainder_cols=False,
            verbose_feature_names_out=False,
            sparse_threshold=0
        )
        preprocessor.fit(df_preprocessed)
        df_preprocessed = pd.DataFrame(preprocessor.transform(df_preprocessed), columns=preprocessor.get_feature_names_out())
        booleans = df_preprocessed.drop(['Age', 'Risk', 'T', 'N', 'M', 'Stage'], axis=1)
        df_preprocessed[booleans.columns] = booleans.astype('int64')
        # one-hot get all, since it does not cause data leakage
        # if test is None: preprocessor.fit(train_preprocessed)
        # preprocessor.fit(np.concat([train_preprocessed.to_numpy(), test_preprocessed.to_numpy()], axis=0))

        # train_preprocessed = pd.DataFrame(preprocessor.transform(train_preprocessed), columns=preprocessor.get_feature_names_out())
        # test_preprocessed = pd.DataFrame(preprocessor.transform(test_preprocessed), columns=preprocessor.get_feature_names_out())
        # booleans = train_preprocessed.drop('Age', axis=1)
        # train_preprocessed[booleans.columns] = booleans.astype('int64')
        # booleans = test_preprocessed.drop('Age', axis=1)
        # test_preprocessed[booleans.columns] = booleans.astype('int64')

    # Scaling
    if do_scale:
        df_preprocessed['Age'] = df_preprocessed['Age'] / 100
    return df_preprocessed
    # if do_scale and type(train_preprocessed) is pd.DataFrame:
    #     train_preprocessed['Age'] = age_scaler.fit_transform(train_preprocessed[['Age']])
    #     test_preprocessed['Age'] = age_scaler.transform(test_preprocessed[['Age']])
    # elif do_scale and type(train_preprocessed) is np.ndarray:
    #     train_preprocessed[:,0:1] = age_scaler.fit_transform(train_preprocessed[:,0:1])
    #     test_preprocessed[:,0:1] = age_scaler.transform(test_preprocessed[:,0:1])
    # return train_preprocessed, test_preprocessed

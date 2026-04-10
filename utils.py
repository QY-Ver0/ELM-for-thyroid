# Some functions used
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, fbeta_score, recall_score, precision_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import MinMaxScaler

from preprocess import preprocess, get_XY

from ELM import ELMclf

def cross_val_score(estimator, X, y=None, scoring=None, cv=None, shuffle=False, stratified=True, args=None, ret_train=False):
    if X is None: raise ValueError('X cannot be None')
    if y is None: raise ValueError('y cannot be None for supervised algorithm')
    if scoring is None: scoring = accuracy_score
    if cv is None: cv = min(5,pd.Series(y).value_counts().min()) # select the minimum possible folds or default of 5
    if pd.Series(y).value_counts().min() == 1 or cv == 1:
        raise ValueError('Cross Validation folds, cv, cannot be 1.')
    if stratified:
        kfold = StratifiedKFold(n_splits=cv, shuffle=shuffle)
    else:
        kfold = KFold(n_splits=cv, shuffle=shuffle)

    train_scores = []
    scores = []
    # get indices of each fold, and run model on each
    for i, (train_index, test_index) in enumerate(kfold.split(X, y)):
        # Fold i:
        x_train = X[train_index].copy()
        y_train = y[train_index].copy()
        x_test  = X[test_index].copy()
        y_test  = y[test_index].copy()
        # if not np.any(x_train[:,0] == 1) and np.any(x_test[:,0] == 1): print(f'Fold {i}: data leakage risk')
        if args:
            estimator.fit(x_train, y_train, **args)
        else:
            estimator.fit(x_train, y_train)
        if ret_train:
            y_pred = estimator.predict(x_train)
            train_scores.append(scoring(y_train, y_pred))
            # print(confusion_matrix(y_train, y_pred))

        y_pred = estimator.predict(x_test)
        scores.append(scoring(y_test, y_pred))
        # print(confusion_matrix(y_test, y_pred))

    if ret_train: return train_scores, scores
    return scores
#
# def cross_val_score(estimator, X, y=None, scoring=None, cv=None, shuffle=False, stratified=True, args=None, random_state=None, ret_train=False):
#     # pass unscaled data in, to ensure data leakage
#     if X is None: raise ValueError('X cannot be None')
#     if y is None: raise ValueError('y cannot be None for supervised algorithm')
#     if scoring is None: scoring = accuracy_score
#     if cv is None: cv = min(5,pd.Series(y).value_counts().min()) # select the minimum possible folds or default of 5
#     if pd.Series(y).value_counts().min() == 1 or cv == 1:
#         raise ValueError('Cross Validation folds, cv, cannot be 1.')
#     if stratified:
#         kfold = StratifiedKFold(n_splits=cv, shuffle=shuffle)
#     else:
#         kfold = KFold(n_splits=cv, shuffle=shuffle)
#
#     train_scores = []
#     scores = []
#
#     X_local = X.to_numpy() if type(X) is pd.DataFrame else X.copy()
#     y_local = y.to_numpy() if type(y) is pd.DataFrame else y.copy()
#     # get indices of each fold, and run model on each
#     for i, (train_index, test_index) in enumerate(kfold.split(X, y)):
#         # Fold i:
#         # x_train = X.iloc[train_index]
#         # y_train = y.iloc[train_index]
#         # x_test  = X.iloc[test_index]
#         # y_test  = y.iloc[test_index]
#         x_train = X_local[train_index]
#         y_train = y_local[train_index]
#         x_test  = X_local[test_index]
#         y_test  = y_local[test_index]
#
#         # do scaling
#         # tr,te = preprocess(
#         #     pd.concat([x_train, y_train], axis=1),
#         #     pd.concat([x_test, y_test], axis=1),
#         #     MinMaxScaler(), do_onehot=False, do_scale=True
#         # )
#         tr,te = preprocess(
#             np.concat([x_train, y_train[:,np.newaxis]], axis=1),
#             np.concat([x_test,  y_test[:, np.newaxis]], axis=1),
#             MinMaxScaler(), do_onehot=False, do_scale=True
#         )
#         x_train, y_train = get_XY(tr, tr.shape[1]-1)
#         x_test,  y_test  = get_XY(te, tr.shape[1]-1)
#         print(np.all(x_train == X_local[train_index]))
#         if args:
#             estimator.fit(x_train, y_train, **args)
#         else:
#             estimator.fit(x_train, y_train)
#
#         y_pred = estimator.predict(x_test)
#         indi_score = (scoring(y_test, y_pred))
#         scores.append(indi_score)
#
#         if ret_train:
#             y_train_pred = estimator.predict(x_train)
#             train_scores.append(scoring(y_train, y_train_pred))
#             # print(confusion_matrix(y_train, y_pred))
#     if ret_train: return train_scores, scores
#     return scores



def get_metrics(y, y_pred):
    # to get pandas dataframe for showing a table of comparison
    # between train result and test result
    # it also considers binary target, since specificity is here
    return {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y, y_pred, average='binary', zero_division=0),
        'f1': f1_score(y, y_pred, average='binary', zero_division=0),
        'f2': fbeta_score(y, y_pred, beta=2, average='binary', zero_division=0),
        'specificity': recall_score(y, y_pred, average='binary', zero_division=0, pos_label=y.min()),
        'npv': precision_score(y, y_pred, average='binary', zero_division=0, pos_label=y.min()),
        'mcc': matthews_corrcoef(y, y_pred),
    }

def get_metrics_df(model_fitted: ELMclf, x_train, x_test, y_train, y_test):
    # Assumes model to be fitted
    if type(x_train) is pd.DataFrame:
        train_pred = model_fitted.predict(x_train.to_numpy())
        test_pred = model_fitted.predict(x_test.to_numpy())
    else:
        train_pred = model_fitted.predict(x_train)
        test_pred = model_fitted.predict(x_test)
    res = []

    for true, pred in [(y_train, train_pred), (y_test, test_pred)]:
        res.append(get_metrics(true,pred))

    metrics_df = pd.DataFrame(res)
    metrics_df = pd.concat((pd.DataFrame(['train_set','test_set'], columns=['predicted_data']), metrics_df), axis=1)

    return metrics_df

def train_test_graph(train_res,test_res):
    if len(train_res) != len(test_res):
        raise Exception('Both train and validate list/array must have same length')
    tt_df = pd.DataFrame([train_res, test_res], index=['Train', 'Validation'])
    tt_df = tt_df.transpose()
    iters = range(1, tt_df.shape[0]+1)
    tt_df.insert(0, 'Iters', iters)

    ax = tt_df.plot.line('Iters', 'Train', color='orange')
    tt_df.plot.line('Iters', 'Validation', color='cyan', ax=ax)
    return ax, tt_df
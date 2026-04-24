# Some functions used
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, fbeta_score, recall_score, precision_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler

from preprocess import preprocess, get_XY

from ELM import ELMclf
def f1(y_true, y_pred):
    raw_y_true = np.asarray(y_true).ravel()
    raw_y_pred = np.asarray(y_pred).ravel()

    # self.raw_y_true = y_true
    # self.raw_y_pred = y_pred

    y_true = np.where(raw_y_true == np.min(raw_y_true), -1, 1)
    y_pred = np.where(raw_y_pred == np.min(raw_y_pred), -1, 1)

    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == -1) & (y_pred == -1))
    FP = np.sum((y_true == -1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == -1))

    return 2 * TP / (2 * TP + FP + FN) if (TP + FP + FN) > 0 else 0
def mcc(y_true, y_pred):
    raw_y_true = np.asarray(y_true).ravel()
    raw_y_pred = np.asarray(y_pred).ravel()

    y_true = np.where(raw_y_true == np.min(raw_y_true), -1, 1)
    y_pred = np.where(raw_y_pred == np.min(raw_y_pred), -1, 1)

    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == -1) & (y_pred == -1))
    FP = np.sum((y_true == -1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == -1))

    return (TP*TN - FP*FN) / ((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) ** 0.5
def cross_val_score(estimator, X, y=None,
                    scoring=None, cv=None,
                    shuffle=False, stratified=True,
                    random_state=None,
                    args=None,
                    ret_train=False, ret_graph_df=False,
                    val_size: float=None):
    # random state will be used in train_test_split if it is enabled
    if X is None: raise ValueError('X cannot be None')
    if y is None: raise ValueError('y cannot be None for supervised algorithm')
    if scoring is None: scoring = accuracy_score
    # if cv is None: cv = min(5,pd.Series(y).value_counts().min()) # select the minimum possible folds or default of 5
    # if pd.Series(y).value_counts().min() == 1 or cv == 1:
    #     raise ValueError('Cross Validation folds, cv, cannot be 1.')
    if stratified:
        if shuffle and random_state:
            kfold = StratifiedKFold(n_splits=cv, shuffle=shuffle, random_state=random_state)
        else:
            kfold = StratifiedKFold(n_splits=cv, shuffle=shuffle)
    else:
        if shuffle and random_state:
            kfold = KFold(n_splits=cv, shuffle=shuffle, random_state=random_state)
        else:
            kfold = KFold(n_splits=cv, shuffle=shuffle)
    X = X.view()
    y = y.ravel()
    train_scores = []
    scores = []
    dflist = []
    scores_dflist = []
    # get indices of each fold, and run model on each
    for i, (train_index, test_index) in enumerate(kfold.split(X, y)):
        # Fold i:
        x_train = X[train_index]
        y_train = y[train_index]
        x_test  = X[test_index]
        y_test  = y[test_index]

        # for abc to work here too
        x_train_sub, x_validate, y_train_sub, y_validate = None, None, None, None
        if val_size is not None:
            # if cv is not 1, validate can just be same as test
            # since it is not used in optimisation at all if cv != 1
            if args is not None and 'cv' in args and args['cv'] != 1:
                x_train_sub, x_validate, y_train_sub, y_validate = x_train, x_test, y_train, y_test
            else:
                x_train_sub, x_validate, y_train_sub, y_validate = train_test_split(x_train, y_train, test_size=val_size, random_state=random_state, stratify=y_train)

        if args:
            if val_size is not None:
                print(f'Fold {i+1}:')
                best_param = estimator.fit(x_train_sub, y_train_sub, x_validate, y_validate, **args)
            else:
                best_param = estimator.fit(x_train, y_train, **args)
        else:
            if val_size is not None:
                best_param = estimator.fit(x_train_sub, y_train_sub, x_validate, y_validate)
            else:
                best_param = estimator.fit(x_train, y_train)

        y_pred = estimator.predict(x_test)
        scores.append(scoring(y_test, y_pred))
        if ret_train:
            y_pred = estimator.predict(x_train)
            train_scores.append(scoring(y_train, y_pred))
            # print(confusion_matrix(y_train, y_pred))


        # print(confusion_matrix(y_test, y_pred))

        # for optimisers
        if ret_graph_df:
            train_res_i = estimator.train_res
            vali_res    = estimator.vali_res
            test_res    = estimator.test_res
            scouts_cnt  = estimator.lim_reached_cnts

            # append DataFrame list
            tt_df = pd.DataFrame([train_res_i, vali_res, test_res, scouts_cnt], index=['Train', 'Validation', 'Test', 'ScoutCall'])
            tt_df = tt_df.transpose()
            iters = range(1, tt_df.shape[0] + 1)
            tt_df.insert(0, 'Iters', iters)
            tt_df.insert(0, 'Fold', np.repeat(i + 1, tt_df.shape[0]))
            dflist.append(tt_df)

            scores_i_df = get_metrics_df(estimator.models[estimator.best_idx].fit(x_train, y_train, **best_param),
                                         x_train, x_test, y_train, y_test)
            scores_i_df.insert(0, 'Fold', np.repeat(i + 1, scores_i_df.shape[0]))
            scores_dflist.append(scores_i_df)

    if ret_train:
        if ret_graph_df: return train_scores, scores, dflist, scores_dflist
        else:            return train_scores, scores
    if ret_graph_df: return dflist, scores_dflist
    else:            return scores
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

def train_test_graph(train_res,test_res,scout_call):
    if len(train_res) != len(test_res):
        raise Exception('Both train and validate list/array must have same length')
    tt_df = pd.DataFrame([train_res, test_res, scout_call], index=['Train', 'Validation', 'ScoutCall'])
    tt_df = tt_df.transpose()
    iters = range(1, tt_df.shape[0]+1)
    tt_df.insert(0, 'Iters', iters)

    fig, (ax_up, ax_down) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4,1]})
    tt_df.plot.line('Iters', 'Train', color='orange', ax=ax_up)
    tt_df.plot.line('Iters', 'Validation', color='cyan', ax=ax_up)
    tt_df.plot.line('Iters', 'ScoutCall', color='grey', ax=ax_down)

    ax_up.set_xlabel('')
    ax_up.tick_params(axis='x', labelbottom=False)
    # plt.fill_between(tt_df[''])

    ax_down.minorticks_on()
    ax_down.grid(which='minor', axis='y', linestyle='--', linewidth=0.4)
    fig.tight_layout()
    return fig, tt_df

def train_test_graph_multiseed_2(tt_df, use_test=False):
    # average of variance of folds for ech model
    group = tt_df.groupby(['Seed', 'Iters'], as_index=False)
    avg_seed_df = group.mean()
    std_seed_df = group.std()
    avg_sub_std_df = avg_seed_df.drop(['Seed', 'Iters'], axis=1) - std_seed_df.drop(['Seed', 'Iters'],                                                                     axis=1)  # In each iters find lower bound per fold
    avg_sub_std_df = pd.concat([avg_seed_df[['Seed', 'Iters']], avg_sub_std_df], axis=1)
    avg_df = avg_sub_std_df.groupby(['Iters'], as_index=False).mean() # avg of models
    std_df = avg_sub_std_df.groupby(['Iters'], as_index=False).std()  # std of that
    fig, (ax_up, ax_down) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]})

    valname = 'Test' if use_test else 'Validation'

    avg_df.plot.line('Iters', 'Train', color='orange', ax=ax_up)
    avg_df.plot.line('Iters', valname, color='cyan', ax=ax_up)
    avg_df.plot.line('Iters', 'ScoutCall', color='grey', ax=ax_down)

    ax_up.set_xlabel('')
    ax_up.tick_params(axis='x', labelbottom=False)
    ax_up.fill_between(avg_df['Iters'],
                       avg_df['Train'] - std_df['Train'],
                       avg_df['Train'] + std_df['Train'],
                       alpha=0.2, color='orange', label='Train_std')
    ax_up.fill_between(avg_df['Iters'],
                       avg_df[valname] - std_df[valname],
                       avg_df[valname] + std_df[valname],
                       alpha=0.2, color='cyan', label='Validation_std')

    ax_down.minorticks_on()
    ax_down.grid(which='minor', axis='y', linestyle='--', linewidth=0.4)

    print(f'Final Validation Std: {std_df.iloc[-1][valname]}')
    print(f'Final Train Std: {std_df.iloc[-1]['Train']}')
    fig.tight_layout()
    # fig.legend()
    return fig, avg_df
def train_test_graph_multiseed(tt_df, use_test=False):
    # average of variance per fold of performance in each model (seed)
    group = tt_df.groupby(['Fold', 'Iters'], as_index=False)
    avg_fold_df = group.mean()
    std_fold_df = group.std()
    return train_test_graph_2(avg_fold_df, std_fold_df, use_test)
def train_test_graph_2(avg_fold_df, std_fold_df, use_test=False):
    avg_df = avg_fold_df.groupby(['Iters'], as_index=False).mean().drop(['Seed'], axis=1)
    # std_df = std_fold_df.groupby(['Iters'], as_index=False).mean().drop(['Seed'], axis=1)
    std_df = avg_fold_df.groupby(['Iters'], as_index=False).std()

    valname = 'Test' if use_test else 'Validation'
    # if use_test: avg_df.rename(columns={'Test': 'Validation'}, inplace=True)
    # valname = 'Validation'

    fig, (ax_up, ax_down) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]})
    avg_df.plot.line('Iters', 'Train', color='orange', ax=ax_up)
    avg_df.plot.line('Iters', valname, color='cyan', ax=ax_up)
    avg_df.plot.line('Iters', 'ScoutCall', color='grey', ax=ax_down)

    ax_up.set_xlabel('')
    ax_up.tick_params(axis='x', labelbottom=False)
    ax_up.fill_between(avg_df['Iters'],
                       avg_df['Train'] - std_df['Train'],
                       avg_df['Train'] + std_df['Train'],
                       alpha=0.2, color='orange', label='Train_std')
    ax_up.fill_between(avg_df['Iters'],
                       avg_df[valname] - std_df[valname],
                       avg_df[valname] + std_df[valname],
                       alpha=0.2, color='cyan', label='Validation_std')

    ax_down.minorticks_on()
    ax_down.grid(which='minor', axis='y', linestyle='--', linewidth=0.4)
    print(f'Final Validation Std: {std_df.iloc[-1][valname]}')
    print(f'Final Train Std: {std_df.iloc[-1]['Train']}')
    fig.tight_layout()
    return fig
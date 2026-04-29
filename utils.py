# Some functions used
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import accuracy_score, f1_score, fbeta_score, recall_score, precision_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from joblib import Parallel, delayed

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
                    val_size: float=None, job=1):
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


    if job > 1:
        print(f'Run CV of job {job}')
        res = Parallel(n_jobs=job)(delayed(indi_loop)(
            estimator, i, X[train_index], y[train_index], X[test_index], y[test_index],
            scoring=scoring,
            random_state=random_state,
            args=args,
            ret_train=ret_train, ret_graph_df=ret_graph_df,
            val_size = val_size
        ) for i, (train_index, test_index) in enumerate(kfold.split(X, y)))
    else:
        res = []
        for i, (train_index, test_index) in enumerate(kfold.split(X, y)):
            # Fold i:
            x_train = X[train_index]
            y_train = y[train_index]
            x_test  = X[test_index]
            y_test  = y[test_index]

            train_score, test_score, dflist, scores_dflist = indi_loop(
                estimator, i, x_train, y_train, x_test, y_test,
                      scoring,
                      random_state, args,
                      ret_train, ret_graph_df,
                      val_size)
            res.append((train_score, test_score, dflist, scores_dflist))
    train_scores, scores, dflist, scores_dflist = tuple(map(list, zip(*res)))

        # if ret_train:
        #     train_scores.append(train_score)

        # # for abc to work here too
        # x_train_sub, x_validate, y_train_sub, y_validate = None, None, None, None
        # if val_size is not None:
        #     # if cv is not 1, validate can just be same as test
        #     # since it is not used in optimisation at all if cv != 1
        #     if args is not None and 'cv' in args and args['cv'] != 1:
        #         x_train_sub, x_validate, y_train_sub, y_validate = x_train, x_test, y_train, y_test
        #     else:
        #         x_train_sub, x_validate, y_train_sub, y_validate = train_test_split(x_train, y_train, test_size=val_size, random_state=random_state, stratify=y_train)
        #
        # if args:
        #     if val_size is not None:
        #         print(f'Fold {i+1}:')
        #         best_param = estimator.fit(x_train_sub, y_train_sub, x_validate, y_validate, **args)
        #     else:
        #         best_param = estimator.fit(x_train, y_train, **args)
        # else:
        #     if val_size is not None:
        #         best_param = estimator.fit(x_train_sub, y_train_sub, x_validate, y_validate)
        #     else:
        #         best_param = estimator.fit(x_train, y_train)
        #
        # y_pred_test = estimator.predict(x_test)
        # if ret_train:
        #     y_pred_train = estimator.predict(x_train)
        #     train_scores.append(scoring(y_train, y_pred_train))
        #     # print(confusion_matrix(y_train, y_pred))
        #
        # scores.append(scoring(y_test, y_pred_test))
        #
        # # for optimisers
        # if ret_graph_df:
        #     train_res_i = estimator.train_res
        #     vali_res    = estimator.vali_res
        #     test_res    = estimator.test_res
        #     scouts_cnt  = estimator.lim_reached_cnts
        #
        #     # append DataFrame list
        #     tt_df = pd.DataFrame([train_res_i, vali_res, test_res, scouts_cnt], index=['Train', 'Validation', 'Test', 'ScoutCall'])
        #     tt_df = tt_df.transpose()
        #     iters = range(1, tt_df.shape[0] + 1)
        #     tt_df.insert(0, 'Iters', iters)
        #     tt_df.insert(0, 'Fold', np.repeat(i + 1, tt_df.shape[0]))
        #     dflist.append(tt_df)
        #
        #     scores_i_df = get_metrics_df(estimator.models[estimator.best_idx].fit(x_train, y_train, **best_param),
        #                                  x_train, x_test, y_train, y_test)
        #     scores_i_df.insert(0, 'Fold', np.repeat(i + 1, scores_i_df.shape[0]))
        #     scores_dflist.append(scores_i_df)

    if ret_train:
        if ret_graph_df: return train_scores, scores, dflist, scores_dflist
        else:            return train_scores, scores
    if ret_graph_df: return dflist, scores_dflist
    else:            return scores

def indi_loop(estimator, i, x_train, y_train, x_test, y_test,
                    scoring=None,
                    random_state=None,
                    args=None,
                    ret_train=False, ret_graph_df=False,
                    val_size: float=None):
    # Fold i:
    # for abc to work here too
    x_train_sub, x_validate, y_train_sub, y_validate = None, None, None, None
    if val_size is not None:
        # if cv is not 1, validate can just be same as test
        # since it is not used in optimisation at all if cv != 1
        if args is not None and 'cv' in args and args['cv'] != 1:
            x_train_sub, x_validate, y_train_sub, y_validate = x_train, x_test, y_train, y_test
        else:
            x_train_sub, x_validate, y_train_sub, y_validate = train_test_split(x_train, y_train, test_size=val_size,
                                                                                random_state=random_state,
                                                                                stratify=y_train)

    if args:
        if val_size is not None:
            print(f'Fold {i + 1}:')
            best_param = estimator.fit(x_train_sub, y_train_sub, x_validate, y_validate, **args)
        else:
            best_param = estimator.fit(x_train, y_train, **args)
    else:
        if val_size is not None:
            best_param = estimator.fit(x_train_sub, y_train_sub, x_validate, y_validate)
        else:
            best_param = estimator.fit(x_train, y_train)

    y_pred_test = estimator.predict(x_test)
    test_score = scoring(y_test, y_pred_test)
    if ret_train:
        y_pred_train = estimator.predict(x_train)
        train_score = scoring(y_train, y_pred_train)
    else:
        train_score = None
    # for optimisers
    if ret_graph_df:
        train_res_i = estimator.train_res
        vali_res = estimator.vali_res
        test_res = estimator.test_res
        scouts_cnt = estimator.lim_reached_cnts

        # append DataFrame list
        tt_df = pd.DataFrame([train_res_i, vali_res, test_res, scouts_cnt],
                             index=['Train', 'Validation', 'Test', 'ScoutCall'])
        tt_df = tt_df.transpose()
        iters = range(1, tt_df.shape[0] + 1)
        tt_df.insert(0, 'Iters', iters)
        tt_df.insert(0, 'Fold', np.repeat(i + 1, tt_df.shape[0]))

        scores_i_df = get_metrics_df(estimator.models[estimator.best_idx].fit(x_train, y_train, **best_param),
                                     x_train, x_test, y_train, y_test)
        scores_i_df.insert(0, 'Fold', np.repeat(i + 1, scores_i_df.shape[0]))
    else:
        tt_df = None
        scores_i_df = None

    return train_score, test_score, tt_df, scores_i_df
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
def train_test_graph_multiseed(
        tt_df, use_test=False, first_avg_fold=False,
        xaxis_scale=None, yaxis_scale=None, y_lim=None, ylabel=None,
        title=None, ax_up=None, ax_down=None, no_scoutcnt=False):
    # NOT first_avg_fold: average of variance per fold of performance in each model (seed)
    # first_avg_fold    : average of variance per seed of performance in different data (fold)
    if not first_avg_fold:
        group = tt_df.groupby(['Fold', 'Iters'], as_index=False)
        avg_fold_df = group.mean()
        std_fold_df = group.std()
        avg_sub_std_df = avg_fold_df
        std_df = std_fold_df
        # return train_test_graph_2(avg_fold_df, std_fold_df, use_test, )
    else:
        group = tt_df.groupby(['Seed', 'Iters'], as_index=False)
        avg_seed_df = group.mean()
        std_seed_df = group.std()
        avg_sub_std_df = avg_seed_df
        # avg_sub_std_df = (
        #         avg_seed_df.drop(['Seed', 'Iters'], axis=1) -
        #         std_seed_df.drop(['Seed', 'Iters'],axis=1))  # In each iters find lower bound per fold
        # avg_sub_std_df = pd.concat([avg_seed_df[['Seed', 'Iters']], avg_sub_std_df], axis=1)
        # avg_sub_std_df['ScoutCall'] = avg_seed_df['ScoutCall']
        std_df = std_seed_df
    return train_test_graph_2(
        avg_sub_std_df, std_df, use_test,
        xaxis_scale, yaxis_scale, y_lim, ylabel, title, ax_up, ax_down, no_scoutcnt)
def train_test_graph_2(
        avg_fold_df, std_fold_df, use_test=False,
        xaxis_scale=None, yaxis_scale=None, y_lim=None, ylabel=None,
        title=None, ax_up=None, ax_down=None, no_scoutcnt=False):
    avg_df = avg_fold_df.groupby(['Iters'], as_index=False).mean().drop(['Seed'], axis=1)
    # std_df = std_fold_df.groupby(['Iters'], as_index=False).mean().drop(['Seed'], axis=1)
    std_df = avg_fold_df.groupby(['Iters'], as_index=False).std()

    upd_vali_name = 'Fitness'
    upd_test_name = 'Validation'
    upd_scout_name = 'Scout Conversions'
    for_plot = avg_df.copy()
    for_plot = for_plot.rename(columns={
        'Validation': upd_vali_name,
        'Test': upd_test_name,
        'ScoutCall': upd_scout_name})

    train_color = 'orange'
    vali_color  = 'dimgrey'
    test_color  = 'cornflowerblue'
    scout_color = 'grey'

    fig = None
    row = 1 if no_scoutcnt else 2
    if ax_up is None or ax_down is None:
        if no_scoutcnt:
            fig, ax_up = plt.subplots(row, 1)
        else:
            fig, (ax_up, ax_down) = plt.subplots(row, 1, gridspec_kw={'height_ratios': [4, 1]})

    for_plot.plot.line('Iters', 'Train', color=train_color, ax=ax_up)
    for_plot.plot.line('Iters', upd_vali_name, color=vali_color, ax=ax_up)
    if use_test: for_plot.plot.line('Iters', upd_test_name, color=test_color, ax=ax_up)
    if not no_scoutcnt: for_plot.plot.line('Iters', upd_scout_name, color=scout_color, ax=ax_down)

    ax_up.fill_between(avg_df['Iters'],
                       avg_df['Train'] - std_df['Train'],
                       avg_df['Train'] + std_df['Train'],
                       alpha=0.2, color=train_color, label='Train_std')
    ax_up.fill_between(avg_df['Iters'],
                       avg_df['Validation'] - std_df['Validation'],
                       avg_df['Validation'] + std_df['Validation'],
                       alpha=0.2, color=vali_color, label=f'{upd_vali_name}_Std')
    if use_test: ax_up.fill_between(avg_df['Iters'],
                       avg_df['Test'] - std_df['Test'],
                       avg_df['Test'] + std_df['Test'],
                       alpha=0.2, color=test_color, label=f'{upd_test_name}_Std')


    ax_up.margins(x=0.01, y=0.05) # top margin for legend
    ax_up.set_ylabel(ylabel, fontweight='bold')

    if xaxis_scale is not None:
        if xaxis_scale[0] is not None: ax_up.xaxis.set_major_locator(MultipleLocator(xaxis_scale[0]))
        if xaxis_scale[1] is not None: ax_up.xaxis.set_minor_locator(MultipleLocator(xaxis_scale[1]))
    if yaxis_scale is not None:
        if yaxis_scale[0] is not None: ax_up.yaxis.set_major_locator(MultipleLocator(yaxis_scale[0]))
        if yaxis_scale[1] is not None: ax_up.yaxis.set_minor_locator(MultipleLocator(yaxis_scale[1]))

    if y_lim is not None:
        ax_up.set_ylim(*y_lim)
    ax_up.grid(which='major', linestyle=':', linewidth=0.5)
    ax_up.spines['top'].set_visible(False)
    ax_up.spines['right'].set_visible(False)

    if not no_scoutcnt:
        ax_up.set_xlabel('')
        ax_up.tick_params(axis='x', labelbottom=False)
    ax_up.set_xlabel(ax_up.get_xlabel(), fontweight='bold')
    ax_up.set_title(title, pad=30)
    if not no_scoutcnt:
        ax_down.margins(x=0.01, y=0.01)
        ax_down.set_xlabel(ax_down.get_xlabel(), fontweight='bold')
        ax_down.set_ylabel('Avg Cnt', fontweight='bold')
        ax_down.minorticks_on()
        ax_down.grid(which='major', axis='x', linestyle='-', linewidth=0.5)
        ax_down.grid(which='minor', axis='y', linestyle='--', linewidth=0.3)
        ax_down.grid(which='major', axis='y', linestyle='-', linewidth=0.5)
    if use_test: print(f'Final Test Std: {std_df.iloc[-1]['Test']}')
    print(f'Final Validation Std: {std_df.iloc[-1]['Validation']}')
    print(f'Final Train Std: {std_df.iloc[-1]['Train']}')
    if use_test: print(f'Final Lower Test: {avg_df.iloc[-1]['Test']-std_df.iloc[-1]['Test']}')
    print(f'Final Lower Vali: {avg_df.iloc[-1]['Validation']-std_df.iloc[-1]['Validation']}')
    print(f'Final Lower Train: {avg_df.iloc[-1]['Train']-std_df.iloc[-1]['Train']}')
    print(f'Final Train-Vali: {avg_df.iloc[-1]['Train']-avg_df.iloc[-1]['Validation']}')
    if fig is not None: fig.tight_layout()

    legend_param = dict(
        fontsize='x-small',
        # bbox_to_anchor=(1.05, 1),
        # loc='best',
    )
    up_param = legend_param | dict(
        ncol=3,
        bbox_to_anchor=(0.47,0.96),
        bbox_transform=ax_up.transAxes,
    )
    down_param = legend_param
    ax_up.legend(**up_param)
    if not no_scoutcnt: ax_down.legend(**down_param)
    if not no_scoutcnt: return fig, (ax_up, ax_down)
    return fig, ax_up

def format_metric_df(
        df, drop=None, rename_dict=None):
    if drop is None:
        drop = ['f2', 'mcc']
    if rename_dict is None:
        rename_dict = {
            'precision'  : 'PPV',
            'recall'     : 'TPR',
            'npv'        : 'NPV',
            'f1'         : 'F1',
            'specificity': 'TNR'}
    return (df.drop(drop, axis=1)
     .rename(columns=rename_dict))
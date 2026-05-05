import copy

import numpy as np
from ELM import ELMclf
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocess import preprocess, get_data
from optimiser_main import Optimiser, ABCOptimiser
from utils import cross_val_score, f1

def multi_seed_cross_val_abc(optimiser: Optimiser, X, Y, args, cv_scoring=f1, cv=5, split_seed=42, ret_graph_df=True, val_size=0.125, seed_range=range(0,102,2), job=1):
    # val_size: validation size for split in train set of validation set
    ret_train = True
    optimiser_copy = copy.deepcopy(optimiser)

    final_df = None
    final_scores_df = None
    for i in seed_range:
        print(f'Seed {i}:')
        optimiser_copy.random_state = i
        train_scores, scores, dflist, scores_dflist = cross_val_score(
            optimiser_copy, X, Y,
            scoring=cv_scoring, cv=cv, args=args,
            random_state=split_seed,
            ret_train=ret_train, ret_graph_df=ret_graph_df,
            val_size=val_size,
            job=job
        )
        full_dflist = pd.concat(dflist)
        full_scores_dflist = pd.concat(scores_dflist)

        print(full_dflist.shape, len(dflist))

        full_dflist.insert(0, 'Seed', np.repeat(i, full_dflist.shape[0]))
        full_scores_dflist.insert(0, 'Seed', np.repeat(i, full_scores_dflist.shape[0]))
        if final_df is None and final_scores_df is None:
            final_df = full_dflist
            final_scores_df = full_scores_dflist
        else:
            final_df = pd.concat([final_df, full_dflist]).reset_index(drop=True)
            final_scores_df = pd.concat([final_scores_df, full_scores_dflist]).reset_index(drop=True)
    return final_df, final_scores_df


class OptimisationParamExecution:
    def __init__(self, n, l2, def_rng_seed, split_seed, cv, path_directory):
        # setting params
        self.n = n
        self.l2 = l2
        self.def_rng_seed = def_rng_seed
        self.split_seed = split_seed
        self.global_rng = np.random.default_rng(def_rng_seed)

        self.defined_fn = f1
        self.cv = cv
        self.path = path_directory

    def fit(self, solution_cnt, max_iter, trial_limit, employed_legacy, onlooker_legacy):
        n = self.n
        l2 = self.l2
        def_rng_seed = self.def_rng_seed
        split_seed = self.split_seed

        defined_fn = self.defined_fn
        cv = self.cv
        global_rng = self.global_rng

        df = get_data('../data/thyroid_diff.csv')
        df = preprocess(df, do_onehot=True, do_scale=True)

        X, Y = df.drop('Recurred_Yes', axis=1), df['Recurred_Yes']
        x_train, x_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2, random_state=split_seed, stratify=Y)
        x_train_sub, x_validate, y_train_sub, y_validate = train_test_split(
            x_train, y_train, test_size=0.25,random_state=split_seed, stratify=y_train)


        elm = ELMclf(n, l2_param=l2, random_state=def_rng_seed)
        elm.fit(x_train_sub.to_numpy(), y_train_sub.to_numpy())

        popu_temp = elm.return_args()  # template
        popu_temp.pop('l2_param')
        popu_range = {'weights_i': (-1, 1), 'biases': (-1, 1)}  # range
        popu = Optimiser.population_generation(solution_cnt, popu_temp, popu_range,
                                               random_state=int(global_rng.uniform(0, 4294967295)))
        abc_args = {
            'population': popu,
            'population_limit': popu_range,
            'max_iter': max_iter,
            'trial_limit': trial_limit,
            'cv': cv,
            'employed_use_legacy': employed_legacy,
            'onlooker_use_legacy': onlooker_legacy,
        }

        abc = ABCOptimiser(elm, fitness_fn=defined_fn, random_state=def_rng_seed)
        abc.set_verbose(True)
        full_df_seed, final_scores_seed = multi_seed_cross_val_abc(
            abc, x_train.to_numpy(), y_train.to_numpy(), abc_args, cv_scoring=defined_fn,
            seed_range=range(0, 10), job=5)

        # potentially dangerous as it may allow path traversal
        full_df_seed.to_csv(
            f'{self.path}/abc_final_n{int(employed_legacy)}{int(onlooker_legacy)}_sol{solution_cnt}_iter{max_iter}_lim{trial_limit}.csv')
        final_scores_seed.to_csv(
            f'{self.path}/abc_scores_final_n{int(employed_legacy)}{int(onlooker_legacy)}_sol{solution_cnt}_iter{max_iter}_lim{trial_limit}.csv')
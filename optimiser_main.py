import copy
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import math
import time

from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler

from utils import cross_val_score, get_metrics_df

def get_dims(param_dict: dict[str, np.ndarray]) -> dict[str, tuple]:
    """
    Note: assumes all value in ndarray (whether it is single value)
    :return: dictionary containing shape information (tuples) of each ndarray, important for recovery
    """
    return {key: value.shape for key, value in param_dict.items()}
def as_solution(param_dict: dict[str, np.ndarray]) -> np.ndarray:
    """
    Concatenates the ndarray(s) (multi-dimension) together into a 1D array,
    one after another, according to dict order
    (a.k.a. encoding from dict to new format)
    :return: a 1D ndarray
    """
    tup = tuple(map(lambda x: x.ravel(), param_dict.values()))
    return np.concatenate(tup).ravel() # axis=0
def as_params(solution_arr: np.ndarray, param_dims: dict[str, tuple]) -> dict[str, np.ndarray]:
    solution_arr = solution_arr.ravel()
    # test whether size will match
    param_dims_flattened = {key: math.prod(shape_tup) for key, shape_tup in param_dims.items()}
    if sum(param_dims_flattened.values()) != solution_arr.size:
        raise Exception('Dimension specified in param_dims do not match solution_arr total elements')

    # Make value cumulative, and represents index
    param_idx = {}
    prev_key = None
    for i, (key, value) in enumerate(param_dims_flattened.items()):
        if prev_key is None:
            param_idx[key] = (0, value)
            prev_key = key
            continue

        # get prev ending point
        prev_value = param_idx[prev_key][1]
        param_idx[key] = (prev_value, prev_value + value) # to become cumulative
        prev_key = key

    # get from solution_arr, then reshape based on the size
    return {key: solution_arr[idx[0]: idx[1]].ravel().reshape(param_dims[key]) for key, idx in param_idx.items()}

# not rlly used cuz only one optimiser here
class OptimiserName(Enum):
    ABC = 1

class Optimiser:
    # def __init__(self, models:list, x_train:np.ndarray, y_train:np.ndarray, x_test:np.ndarray, y_test:np.ndarray, fitness_fn =accuracy_score, random_state:int=0):
    def __init__(self, models:list, fitness_fn=accuracy_score, random_state:int=0):
        self.best_individual = None # The best param settings
        self.best_idx        = None
        self.models          = models

        self.x_train = None # can be overwritten
        self.y_train = None # can be overwritten
        self.x_test  = None # can be overwritten
        self.y_test  = None # can be overwritten
        self.x       = None
        self.y       = None # unused now

        self.fitness_fn   = fitness_fn
        self.random_state = random_state

        self.train_res = []
        self.vali_res  = [] # or validation, depends on what is obtained
        self.test_res  = [] # validation is for cv_validation, test is for validation
        # when cv_on_train False, vali and test is same

        self.cv_on_train = False
    def predict(self, x):
        x = x.view()
        return self.models[self.best_idx].predict(x)
    def compute_fitness(self, args : dict, index : int = 0, ret_train : bool = False) -> float | tuple[float, float]:
        """

        :param ret_train: return (train_score, test_score) if True
        :param args: dict of params to pass to fit function after x and y
        :param index: index of models (now uses multiple)
        :return: score
        """
        self.models[index].fit(self.x_train,self.y_train,**args)
        tr_score = self.fitness_fn(self.y_train, self.models[index].predict(self.x_train))
        te_score = self.fitness_fn(self.y_test, self.models[index].predict(self.x_test))
        if ret_train: return tr_score, te_score
        return te_score
    def cross_eval(self, args : dict, index : int = 0, cv: int = 5, ret_train : bool = False) -> np.floating | tuple[np.floating, np.floating]:
        if not ret_train:
            if self.cv_on_train:
                return np.mean(cross_val_score(
                    self.models[index],
                    self.x_train,
                    self.y_train,
                    scoring=self.fitness_fn, cv=cv, args=args, ret_train=ret_train))
            else:
                # THIS SHOULD NOT BE USED, this only stays for legacy purposes
                return np.mean(cross_val_score(
                    self.models[index],
                    self.x,
                    self.y,
                    scoring=self.fitness_fn, cv=cv, args=args, ret_train=ret_train))
        if self.cv_on_train:
            # then test is actually validation score within the training
            train, test = cross_val_score(
                self.models[index],
                self.x_train,
                self.y_train,
                scoring=self.fitness_fn, cv=cv, args=args, ret_train=ret_train)
        else:
            # THIS SHOULD NOT BE USED, this only stays for legacy purposes
            train, test = cross_val_score(
                    self.models[index],
                    self.x,
                    self.y,
                    scoring=self.fitness_fn, cv=cv, args=args, ret_train=ret_train)
        return np.mean(train), np.mean(test)
    def score(self):
        if self.best_individual is None or self.best_idx is None:
            raise Exception('Optimiser not fitted')

        final_score = self.compute_fitness(self.best_individual, self.best_idx)
        fn_name = getattr(self.fitness_fn, '__name__', 'Unknown scoring')
        print(f"Final Score ({fn_name}) on Validation Set: {final_score:.4f}")

    def get_best_param(self):
        return self.best_individual

    def get_models(self):
        return self.models

    @staticmethod
    def population_generation(population_size, param_templates, param_ranges, random_state=0):
        """
        :param population_size: amount of randomly generated parameters
        # :param param_templates: param type for each, indicating size (: supporting numpy array and numbers only :)
        :param param_ranges: the example parameters to generate, in dict
        :return:
        """
        population = []
        rng = np.random.default_rng(random_state)
        for _ in range(population_size):
            individual = param_ranges.copy()
            for param in param_ranges:
                min, max = param_ranges[param]
                param_type = type(param_templates[param])
                # rng = np.random.default_rng(random_state)

                if param_type is int or isinstance(param_templates[param], np.integer):
                    individual[param] = rng.integers(min, max)
                elif param_type is float or isinstance(param_templates[param], np.floating):
                    individual[param] = rng.uniform(min, max)
                elif param_type is np.ndarray:
                    elem_example = param_templates[param].flatten()[0]
                    elem_type = param_templates[param].dtype
                    if elem_type is int or isinstance(elem_example, np.integer):
                        individual[param] = rng.integers(min, max, size=param_templates[param].shape)
                    elif elem_type is float or isinstance(elem_example, np.floating):
                        individual[param] = rng.uniform(min, max, size=param_templates[param].shape)
                    else:
                        print('Unidentified value')
                        print(isinstance(elem_type, np.floating))
                        raise TypeError('Unidentified param type in array')
                else:
                    print(type(min))
                    raise TypeError('Unidentified value')
            population.append(individual)
        return population

class ABCOptimiser(Optimiser):
    # def __init__(self, model, x_train, y_train, x_test, y_test, fitness_fn=accuracy_score, random_state=0):
    def __init__(self, model, fitness_fn=accuracy_score, random_state=0):
        # super().__init__([model],x_train,y_train,x_test,y_test,fitness_fn,random_state)
        super().__init__([model],fitness_fn,random_state)
        self.lim_reached_cnts    = None

        self.best_solution       = None
        self.solutions           = np.nan
        self.scores              = np.nan
        self.best_score          = 0
        self.train_score         = 0
        self.test_score          = 0

        # legacy
        self.max_copy            = None
        self.min_copy            = None

        # IWO-inspired
        self.nmi                 = None # nonlinear modulation index, usually 3
        self.max_change          = None
        self.min_change          = None
        self.final_sigma         = None
        self.initial_sigma       = None

        self.max_iter            = None
        self.POPULATION_LIMIT    = None
        self.POPULATION_TEMPLATE = None
        self.PARAM_DIMS          = None
        self.TRIAL_LIMIT         = None
        self.verbose             = False

        self.employed_use_legacy = True
        self.onlooker_use_legacy = True
        # legacy for original, False for IWO-inspired

    def set_verbose(self, to_verbose: bool):
        self.verbose = to_verbose

    def fit(self,
            x_train, y_train, x_test, y_test,
            population: list[dict], population_limit: dict,
            max_iter: int = 1000, trial_limit = 10,
            min_copy = 0, max_copy = 1,
            initial_sigma = 0, final_sigma = 1,
            min_change = 0, max_change = 1,
            nmi = 3,
            employed_use_legacy = True, onlooker_use_legacy = True,
            cv=1, cv_on_train=False):
        self.cv_on_train = cv_on_train
        self.best_individual = None # The best param settings
        self.best_idx        = None

        self.lim_reached_cnts = []
        self.train_res        = []
        self.vali_res         = []
        self.test_res         = [] # or validation, depends on what is obtained
        self.x_train = x_train.view() # can be overwritten
        self.y_train = y_train.ravel() # can be overwritten
        self.x_test  = x_test.view()  # can be overwritten
        self.y_test  = y_test.ravel()  # can be overwritten
        self.x       = np.concat([x_train, x_test])
        self.y       = np.concat([y_train, y_test])
        if len(population) < 1:
            pass

        models = [copy.deepcopy(self.models[0]) for _ in range(len(population))] # len(population) = SN
        self.models = models
        self.best_solution       = None
        self.best_score          = 0
        self.train_score         = 0
        self.max_iter            = max_iter
        self.POPULATION_LIMIT    = population_limit # aka param_ranges
        self.POPULATION_TEMPLATE = population[0]
        self.PARAM_DIMS          = get_dims(population[0])
        self.TRIAL_LIMIT         = trial_limit

        rng = np.random.default_rng(self.random_state)
        f_ss_idx = np.vectorize(lambda i: rng.choice(indexes.ravel()[indexes.ravel() != i]))

        # legacy
        self.max_copy = max_copy
        self.min_copy = min_copy

        # IWO-inspired
        self.nmi                 = nmi # nonlinear modulation index, usually 3
        self.max_change          = max_change
        self.min_change          = min_change
        self.final_sigma         = final_sigma
        self.initial_sigma       = initial_sigma

        self.employed_use_legacy = employed_use_legacy
        self.onlooker_use_legacy = onlooker_use_legacy

        self.solutions = np.concatenate(tuple(map(lambda x: x.reshape(1,-1), map(as_solution, population)))).view()  # of shape (SN, D)

        if cv == 1:
            tr_te_scores = np.array([list(self.compute_fitness(population, i, True)) for i, population in enumerate(population)])
            train_scores = np.array(tr_te_scores[:,0])
            self.scores = np.array(tr_te_scores[:,1])
        else:
            tr_te_scores = np.array([list(self.cross_eval(population, i, cv, True)) for i, population in enumerate(population)])
            train_scores = np.array(tr_te_scores[:,0])
            self.scores = np.array(tr_te_scores[:,1])

        SN, D = self.solutions.shape

        indexes = np.arange(SN)

        total_time = 0.
        trials = np.repeat(0, SN)
        # trials = [0] * SN
        for current_iter in range(max_iter+1):
            # Timing purpose
            start_time = time.time()
            # only use current_iter of 1 to max_iter (inclusive)
            if current_iter == 0: continue
            if self.best_score >= 1:
                # break
                self.train_res.append(self.train_score)
                self.vali_res.append(self.best_score)
                self.test_res.append(self.test_score)
                self.lim_reached_cnts.append(0)

                end_time = time.time()
                total_time += end_time - start_time
                if self.verbose: print(
                    '\r',
                    f'Iter {current_iter}/{max_iter}: {end_time - start_time:.4f} s |',
                    f'Avg time: {total_time / current_iter:.4f} s |',
                    f'Best validation: {self.compute_fitness(self.best_individual, self.best_idx):.4f} |',
                    f'Best cross_val: {self.cross_eval(self.best_individual, self.best_idx, cv=5, ret_train=False):.4f}, idx: {self.best_idx} |',
                    f'Solutions: {len(self.scores.ravel())}',
                    end='')
                continue



            # current p copy
            curr_p_copy = ABCOptimiser.get_p_copy(current_iter, max_iter, self.min_copy, self.max_copy)
            curr_sigma  = ABCOptimiser.get_sigma (current_iter, max_iter, self.initial_sigma, self.final_sigma, self.nmi)

            # Employed bee process, single loop to check whether generated solution are better
            second_solutions_idxes = f_ss_idx(indexes).ravel()
            # second_solutions = np.vectorize(lambda i: self.solutions[rng.choice(indexes[indexes != i])], otypes=[np.ndarray])(indexes)
            second_solutions = self.solutions[second_solutions_idxes]
            vs = [self.neighbourhood_gen(self.solutions[i], second_solutions[i], curr_p_copy, curr_sigma, self.scores.ravel()[i], rng, True) for i in indexes]
            for i in indexes:
                idx = int(i)
                v = vs[idx].ravel()
                # Skip if Si is same as Vi
                # v_solution = as_solution(v)
                v_solution = v
                if np.all(v_solution == self.solutions[i]):
                    trials[i] += 1
                    # print(f'\t{i} Employed: Literally the same')
                    continue
                v_param = as_params(v, self.PARAM_DIMS)
                if cv == 1:
                    v_tr_score, v_score = self.compute_fitness(v_param,idx,True)
                else:
                    v_tr_score, v_score = self.cross_eval(v_param, idx, cv, True)
                # replace Si if Vi is better
                if v_score > self.scores.ravel()[idx]:
                    self.solutions[i] = v_solution
                    self.scores.ravel()[idx] = v_score
                    train_scores[idx] = v_tr_score
                    trials[i] = 0
                else:
                    trials[i] += 1

            fitness_sum = np.sum(self.scores.ravel())
            selection_probability = np.divide(self.scores.ravel(), fitness_sum)
            # Onlooker bee process
            solution_idxes = rng.choice(indexes, size=SN, p=selection_probability).ravel()
            # selected_solutions = self.solutions[solution_idxes].view()
            second_solution_idxes = f_ss_idx(solution_idxes)
            second_solutions = self.solutions[second_solution_idxes]
            vs = [self.neighbourhood_gen(self.solutions[i], second_solutions[i], curr_p_copy, curr_sigma, self.scores.ravel()[i], rng, False) for i in solution_idxes]
            for i in indexes:
                solution_idx = solution_idxes[i]
                v = vs[i].ravel()
                # v_solution = as_solution(v)
                v_solution = v
                if np.all(v_solution == self.solutions[solution_idx]):
                    trials[solution_idx] += 1
                    # print(f'\t{i} Onlooker: Literally the same')
                    continue
                v_param = as_params(v, self.PARAM_DIMS)
                if cv == 1:
                    v_tr_score, v_score = self.compute_fitness(v_param,solution_idx,True)
                else:
                    v_tr_score, v_score = self.cross_eval(v_param, solution_idx, cv, True)
                # Replace Si if Vi is better
                if v_score > self.scores.ravel()[solution_idx]:
                    self.solutions[solution_idx] = v_solution
                    self.scores.ravel()[solution_idx] = v_score
                    train_scores[solution_idx] = v_tr_score
                    trials[solution_idx] = 0
                else:
                    trials[solution_idx] += 1

            # Scout bee process, find new source if score is not improving
            are_lim_reached = trials > (self.TRIAL_LIMIT-1) # a mask
            lim_reached_cnt = np.count_nonzero(are_lim_reached)
            for i in indexes:
                if trials[i] > self.TRIAL_LIMIT-1:
                    # if self.verbose: print(f'\t{i}: Update its solution')
                    self.solutions[i] = as_solution(
                        Optimiser.population_generation(
                            1, self.POPULATION_TEMPLATE, self.POPULATION_LIMIT, int(rng.uniform(1, 4294967295))
                        )[0]
                    )
                    trials[i] = 0

            # Save current best solution
            current_best_idx = np.argmax(self.scores.ravel())
            # if self.verbose: print(f"\tCurrent highest score: {self.scores.ravel()[current_best_idx]}")

            if self.scores.ravel()[current_best_idx] > self.best_score:
                self.best_individual = as_params(self.solutions[current_best_idx], self.PARAM_DIMS)
                self.best_idx = int(current_best_idx)
                # if self.verbose: print(f"\tUpdated score: {self.scores.ravel()[current_best_idx]}, idx: {current_best_idx}")
                # if self.verbose: print(f'\tBest test score : {self.compute_fitness(self.best_individual,self.best_idx)}, idx: {self.best_idx}')
                # if self.verbose: print(f'\tBest cross score: {self.cross_eval(self.best_individual,self.best_idx,cv=cv,ret_train=False)}, idx: {self.best_idx}')

                # t_score = self.compute_fitness(as_params(self.solutions[current_best_idx],self.PARAM_DIMS), int(current_best_idx)) if cv == 1 else self.cross_eval(as_params(self.solutions[current_best_idx], self.PARAM_DIMS), int(current_best_idx), cv, True)[0]
                # self.best_score = copy.deepcopy(self.scores.ravel()[current_best_idx])
                self.best_score = self.scores.ravel()[current_best_idx]
                self.train_score = train_scores[current_best_idx]
                self.test_score = self.compute_fitness(self.best_individual,self.best_idx)

            elif self.best_idx is None:
                # sometimes it just happens
                self.best_individual = as_params(self.solutions[current_best_idx], self.PARAM_DIMS)
                self.best_idx = int(current_best_idx)

            # set to train and test_res
            self.train_res.append(self.train_score)
            self.vali_res.append(self.best_score)
            self.test_res.append(self.test_score)
            self.lim_reached_cnts.append(lim_reached_cnt)

            end_time = time.time()
            total_time += end_time - start_time
            if self.verbose: print(
                '\r',
                f'Iter {current_iter}/{max_iter}: {end_time - start_time:.4f} s |',
                f'Avg time: {total_time / current_iter:.4f} s |',
                f'Best validation: {self.compute_fitness(self.best_individual,self.best_idx):.4f} |',
                f'Best cross_val: {self.cross_eval(self.best_individual, self.best_idx,cv=5,ret_train=False):.4f}, idx: {self.best_idx} |',
                f'Solutions: {len(self.scores.ravel())}',
                end='')


        if self.verbose: print(f'\nBest score recorded: {self.best_score}, at idx: {self.best_idx}, total time: {total_time:.4f} seconds')
        # self.best_individual
        # self.best_idx
        # if self.verbose: print(f'Best score: {self.compute_fitness(self.best_individual,self.best_idx)}, idx: {self.best_idx}, ')
        return self.best_individual
    def neighbourhood_gen(self, solution: np.ndarray, solution_k: np.ndarray, p_copy, sigma, score, rng, is_employed:bool):
        """
        Run with Variable Copy
        :return: dict, not ndarray
        """
        if (is_employed and self.employed_use_legacy) or (not is_employed and self.onlooker_use_legacy):
            return self.neighbourhood_gen_legacy(solution, solution_k, p_copy, rng)
        else:
            return self.neighbourhood_gen_IWO(solution, score, sigma, rng)

    def neighbourhood_gen_legacy(self, solution: np.ndarray, solution_k: np.ndarray, p_copy: int, rng):
        # p_copy = ABCOptimiser.get_p_copy(curr_iter, self.max_iter, self.min_copy, self.max_copy)
        solution = solution.ravel()
        solution_k = solution_k.ravel()
        should_copy = rng.random(solution.shape) < p_copy # array of booleans
        phis = rng.uniform(-1, 1, solution.shape)
        res = np.where(should_copy, solution, solution + phis * (solution - solution_k))
        return np.clip(res, -1, 1) # for now since limits are same, just this
        # clipping based on the limits
        # return ABCOptimiser.clip_to_ranges(as_params(res, self.PARAM_DIMS), self.POPULATION_LIMIT)

    def neighbourhood_gen_IWO(self, solution: np.ndarray, score, sigma, rng):
        solution = solution.ravel()
        D = solution.shape[0]
        change_ratio = ABCOptimiser.get_change_ratio(self.best_score, score, self.min_change, self.max_change)
        change_cnt   = np.ceil(D * change_ratio).astype(int) # original / 100 cuz it is percentage
        change_cnt   = max(1, change_cnt) # small possibility of change even when change_cnt < 1
        gaussian_additive = rng.normal(0, scale=sigma, size=change_cnt)
        indexes = rng.choice(np.arange(D), size=change_cnt, replace=False)
        res = solution.copy()
        res[indexes] = np.clip(solution[indexes] + gaussian_additive, -1, 1)
        return res.ravel()

    @staticmethod
    def clip_to_ranges(params: dict, ranges: dict) -> dict:
        if params.keys() != ranges.keys():
            raise Exception("Param keys are not equal to the range keys given while clipping.")

        return {key: np.clip(params[key], value[0], value[1]) for key, value in ranges.items()}
        
    @staticmethod
    def get_p_copy(curr_iter: int, max_iter: int, min_copy, max_copy):
        """
        Variable Copy
        """
        return (curr_iter / max_iter) * (max_copy - min_copy) + min_copy

    @staticmethod
    def get_sigma(curr_iter: int, max_iter: int, ini_sigma, final_sigma, nmi):
        # difference for this ratio to p_copy is that p_copy increase as iter increase,
        # this decrease as iter increase, and has a nmi to increase its speed of decay

        # ini sigma should be larger than final
        iter_ratio = (max_iter - curr_iter) / max_iter
        iter_ratio = iter_ratio ** nmi

        return iter_ratio * (ini_sigma - final_sigma) + final_sigma

    @staticmethod
    def get_change_ratio(highest_fit, this_fit, min_change, max_change):
        # as fitness difference decrease, change decrease
        # same idea as p_copy but this changes based on fitness of solutions available
        if highest_fit == 0: fit_ratio = 1
        else: fit_ratio = (highest_fit - this_fit) / highest_fit
        return fit_ratio * (max_change - min_change) + min_change

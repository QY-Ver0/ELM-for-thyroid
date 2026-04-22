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

        # self.x_train = x_train # can be overwritten
        # self.y_train = y_train # can be overwritten
        # self.x_test  = x_test  # can be overwritten
        # self.y_test  = y_test  # can be overwritten
        self.x_train = None # can be overwritten
        self.y_train = None # can be overwritten
        self.x_test  = None # can be overwritten
        self.y_test  = None # can be overwritten
        self.x       = None
        self.y       = None

        self.fitness_fn   = fitness_fn
        self.random_state = random_state

        self.train_res = []
        self.test_res  = [] # or validation, depends on what is obtained
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
            return np.mean(cross_val_score(
                self.models[index],
                self.x,
                self.y,
                scoring=self.fitness_fn, cv=cv, args=args, ret_train=ret_train))

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
        self.best_score          = 0
        self.train_score         = 0
        self.max_copy            = None
        self.min_copy            = None
        self.max_iter            = None
        self.POPULATION_LIMIT    = None
        self.POPULATION_TEMPLATE = None
        self.PARAM_DIMS          = None
        self.TRIAL_LIMIT         = None
        self.verbose             = False

        # legacy for original, False for IWO-inspired

    def set_verbose(self, to_verbose: bool):
        self.verbose = to_verbose

    def fit(self, x_train, y_train, x_test, y_test, population: list[dict], population_limit: dict, max_iter: int = 1000, trial_limit = 10, min_copy = 0, max_copy = 1, cv=1):
        self.best_individual = None # The best param settings
        self.best_idx        = None

        self.lim_reached_cnts = []
        self.train_res        = []
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
        self.max_copy            = max_copy
        self.min_copy            = min_copy
        self.max_iter            = max_iter
        self.POPULATION_LIMIT    = population_limit # aka param_ranges
        self.POPULATION_TEMPLATE = population[0]
        self.PARAM_DIMS          = get_dims(population[0])
        self.TRIAL_LIMIT         = trial_limit

        rng = np.random.default_rng(self.random_state)

        solutions = np.concatenate(tuple(map(lambda x: x.reshape(1,-1), map(as_solution, population)))).view()  # of shape (SN, D)
        if cv == 1:
            tr_te_scores = np.array([list(self.compute_fitness(population, i, True)) for i, population in enumerate(population)])
            # train_scores = np.array([tr_te_scores[i][0] for i in range(len(tr_te_scores))])
            # scores = [tr_te_scores[i][1] for i in range(len(tr_te_scores))]
            train_scores = np.array(tr_te_scores[:,0])
            scores = np.array(tr_te_scores[:,1])
            # scores = [self.compute_fitness(population,i) for i,population in enumerate(population)]
        else:
            tr_te_scores = np.array([list(self.cross_eval(population, i, cv, True)) for i, population in enumerate(population)])
            # train_scores = [tr_te_scores[i][0] for i in range(len(tr_te_scores))]
            # scores = [tr_te_scores[i][1] for i in range(len(tr_te_scores))]
            train_scores = np.array(tr_te_scores[:,0])
            scores = np.array(tr_te_scores[:,1])
            # scores = [self.cross_eval(population,i, cv) for i,population in enumerate(population)]
        # scores = list(map(self.compute_fitness, population))
        f_ss_idx = np.vectorize(lambda i: rng.choice(indexes.ravel()[indexes.ravel() != i]))

        SN, D = solutions.shape
        indexes = np.arange(SN)

        total_time = 0.
        trials = np.repeat(0, SN)
        # trials = [0] * SN
        for current_iter in range(max_iter+1):
            # only use current_iter of 1 to max_iter (inclusive)
            if current_iter == 0: continue
            if self.best_score >= 1: break

            # Timing purpose
            start_time = time.time()

            # current p copy
            curr_p_copy = ABCOptimiser.get_p_copy(current_iter, max_iter, self.min_copy, self.max_copy)

            # Employed bee process, single loop to check whether generated solution are better
            second_solutions_idxes = f_ss_idx(indexes).ravel()
            # second_solutions = np.vectorize(lambda i: solutions[rng.choice(indexes[indexes != i])], otypes=[np.ndarray])(indexes)
            second_solutions = solutions[second_solutions_idxes]
            vs = [self.neighbourhood_gen(solutions[i], second_solutions[i], curr_p_copy, rng) for i in indexes]
            for i in indexes:
                idx = int(i)
                v = vs[idx]
                # Skip if Si is same as Vi
                v_solution = as_solution(v)
                if np.all(v_solution == solutions[i]):
                    trials[i] += 1
                    # print(f'\t{i} Employed: Literally the same')
                    continue
                if cv == 1:
                    v_tr_score, v_score = self.compute_fitness(v,idx,True)
                else:
                    v_tr_score, v_score = self.cross_eval(v, idx, cv, True)
                # replace Si if Vi is better
                if v_score > scores[idx]:
                    solutions[i] = v_solution
                    scores[idx] = v_score
                    train_scores[idx] = v_tr_score
                    trials[i] = 0
                else:
                    trials[i] += 1

            fitness_sum = np.sum(scores)
            selection_probability = np.divide(scores, fitness_sum)
            # Onlooker bee process
            solution_idxes = rng.choice(indexes, size=SN, p=selection_probability).ravel()
            # selected_solutions = solutions[solution_idxes].view()
            second_solution_idxes = f_ss_idx(solution_idxes)
            second_solutions = solutions[second_solution_idxes]
            vs = [self.neighbourhood_gen(solutions[i], second_solutions[i], curr_p_copy, rng) for i in solution_idxes]
            for i in indexes:
                solution_idx = solution_idxes[i]
                v = vs[i]
                v_solution = as_solution(v)
                if np.all(v_solution == solutions[solution_idx]):
                    trials[solution_idx] += 1
                    # print(f'\t{i} Onlooker: Literally the same')
                    continue
                if cv == 1:
                    v_tr_score, v_score = self.compute_fitness(v,solution_idx,True)
                else:
                    v_tr_score, v_score = self.cross_eval(v, solution_idx, cv, True)
                # Replace Si if Vi is better
                if v_score > scores[solution_idx]:
                    solutions[solution_idx] = v_solution
                    scores[solution_idx] = v_score
                    train_scores[solution_idx] = v_tr_score
                    trials[solution_idx] = 0
                else:
                    trials[solution_idx] += 1

            # Scout bee process, find new source if score is not improving
            are_lim_reached = trials > (self.TRIAL_LIMIT-1) # a mask
            lim_reached_cnt = np.count_nonzero(are_lim_reached)
            # if lim_reached_cnt > 0:
            #     solutions[are_lim_reached] = Optimiser.population_generation(
            #         lim_reached_cnt, self.POPULATION_TEMPLATE, self.POPULATION_LIMIT, int(rng.uniform(1, 4294967295))
            #     )
            #     print(solutions[are_lim_reached], '\n')
            #     trials[are_lim_reached] = 0
            for i in indexes:
                if trials[i] > self.TRIAL_LIMIT-1:
                    # if self.verbose: print(f'\t{i}: Update its solution')
                    solutions[i] = as_solution(
                        Optimiser.population_generation(
                            1, self.POPULATION_TEMPLATE, self.POPULATION_LIMIT, int(rng.uniform(1, 4294967295))
                        )[0]
                    )
                    trials[i] = 0

            # Save current best solution
            current_best_idx = np.argmax(scores)
            # if self.verbose: print(f"\tCurrent highest score: {scores[current_best_idx]}")

            if scores[current_best_idx] > self.best_score:
                self.best_individual = as_params(solutions[current_best_idx], self.PARAM_DIMS)
                self.best_idx = int(current_best_idx)
                # if self.verbose: print(f"\tUpdated score: {scores[current_best_idx]}, idx: {current_best_idx}")
                # if self.verbose: print(f'\tBest test score : {self.compute_fitness(self.best_individual,self.best_idx)}, idx: {self.best_idx}')
                # if self.verbose: print(f'\tBest cross score: {self.cross_eval(self.best_individual,self.best_idx,cv=cv,ret_train=False)}, idx: {self.best_idx}')

                # t_score = self.compute_fitness(as_params(solutions[current_best_idx],self.PARAM_DIMS), int(current_best_idx)) if cv == 1 else self.cross_eval(as_params(solutions[current_best_idx], self.PARAM_DIMS), int(current_best_idx), cv, True)[0]
                # self.best_score = copy.deepcopy(scores[current_best_idx])
                self.best_score = scores[current_best_idx]
                self.train_score = train_scores[current_best_idx]

            elif self.best_idx is None:
                # sometimes it just happens
                self.best_individual = as_params(solutions[current_best_idx], self.PARAM_DIMS)
                self.best_idx = int(current_best_idx)

            # set to train and test_res
            self.train_res.append(self.train_score)
            self.test_res.append(self.best_score)
            self.lim_reached_cnts.append(lim_reached_cnt)

            end_time = time.time()
            total_time += end_time - start_time
            if self.verbose: print(
                '\r',
                f'Iter {current_iter}/{max_iter}: {end_time - start_time:.4f} s |',
                f'Avg time: {total_time / current_iter:.4f} s |',
                f'Best validation: {self.compute_fitness(self.best_individual,self.best_idx):.4f} |',
                f'Best cross_val: {self.cross_eval(self.best_individual, self.best_idx,cv=5,ret_train=False):.4f}, idx: {self.best_idx} |',
                f'Solutions: {len(scores)}',
                end='')


        if self.verbose: print(f'\nBest score recorded: {self.best_score}, at idx: {self.best_idx}, total time: {total_time:.4f} seconds')
        # self.best_individual
        # self.best_idx
        # if self.verbose: print(f'Best score: {self.compute_fitness(self.best_individual,self.best_idx)}, idx: {self.best_idx}, ')
        return self.best_individual
    def neighbourhood_gen(self, solution: np.ndarray, solution_k: np.ndarray, p_copy: int, rng):
        """
        Run with Variable Copy
        :return: dict, not ndarray
        """
        # p_copy = ABCOptimiser.get_p_copy(curr_iter, self.max_iter, self.min_copy, self.max_copy)
        solution = solution.ravel()
        solution_k = solution_k.ravel()
        should_copy = rng.random(solution.shape) < p_copy # array of booleans
        phis = rng.uniform(-1, 1, solution.shape)
        res = np.where(should_copy, solution, solution + phis * (solution - solution_k))
        # clipping based on the limits
        return ABCOptimiser.clip_to_ranges(as_params(res, self.PARAM_DIMS), self.POPULATION_LIMIT)
    
    
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
        return (max_copy - min_copy) * (curr_iter / max_iter) + min_copy

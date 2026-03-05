import copy

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
import math
import time
# import threading

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
    start = time.time()
    tup = tuple(map(lambda x: x.flatten(), param_dict.values()))
    return np.concatenate(tup) # axis=0
def as_params(solution_arr: np.ndarray, param_dims: dict[str, tuple]) -> dict[str, np.ndarray]:
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
    return {key: solution_arr[idx[0]: idx[1]].reshape(param_dims[key]) for key, idx in param_idx.items()}

class Optimiser:
    def __init__(self, models:list, x_train:np.ndarray, y_train:np.ndarray, x_test:np.ndarray, y_test:np.ndarray, fitness_fn =accuracy_score, random_state:int=0):
        self.best_individual = None # The best param settings
        self.best_idx = None
        self.models = models
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.fitness_fn = fitness_fn
        self.random_state=random_state

    def compute_fitness(self, args, index=0):
        """

        :param args: dict of params to pass to fit function after x and y
        :param index: index of models (now uses multiple)
        :return: score
        """
        self.models[index].fit(self.x_train,self.y_train,**args)
        # return precision_score(self.y_test, self.model.predict(self.x_test), zero_division=0, average='binary') # FIXME
        return self.fitness_fn(self.y_test, self.models[index].predict(self.x_test))

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


class GAOptimiser(Optimiser):
    def __init__(self, model, x_train, y_train, x_test, y_test, fitness_fn=accuracy_score, random_state=0):
        super().__init__([model],x_train,y_train,x_test,y_test,fitness_fn,random_state)

    def generation_loop(self, population, generations_cnt, mutation_rate=0.1):
        best_fitness_history = []
        average_fitness_history = []
        initial_population_size = len(population)
        prev_fitness_scores = []
        rng = np.random.default_rng(self.random_state)

        cnt = 1
        for generation in range(generations_cnt):
            print("Generation: ", cnt)
            cnt += 1
            fitness_scores = np.array([self.compute_fitness(individual) for individual in population])
            best_fitness = np.max(fitness_scores)
            average_fitness = np.mean(fitness_scores)
            best_fitness_history.append(best_fitness)
            average_fitness_history.append(average_fitness)

            # Selection
            sorted_indices = np.argsort(fitness_scores)[::-1]  # desc order
            population = [population[i] for i in sorted_indices[:initial_population_size // 2]]  # desc, top 50% of population

            # Crossover and Mutation
            new_population = []
            while len(new_population) < initial_population_size:
                parents = rng.choice(population, 2, replace=False)
                child={}
                for param in parents[0]:
                    child[param] = (parents[0][param] + parents[1][param]) / 2

                    if rng.random() < mutation_rate:
                        child[param] += rng.random(child[param].shape) * 0.1 # asterisk is unpacking the shape for np.random.rand function
                new_population.append(child)
            population = new_population

            prev_fitness_scores = fitness_scores
        self.best_individual = population[np.argmax(prev_fitness_scores)]
        self.best_idx = 0 # only one model
        return self.best_individual

temp_best = None
class ABCOptimiser(Optimiser):
    def __init__(self, model, x_train, y_train, x_test, y_test, fitness_fn=accuracy_score, random_state=0):
        super().__init__([model],x_train,y_train,x_test,y_test,fitness_fn,random_state)
        self.best_solution = None
        self.best_score = 0
        self.max_copy = None
        self.min_copy = None
        self.max_iter = None
        self.POPULATION_LIMIT = None
        self.POPULATION_TEMPLATE = None
        self.PARAM_DIMS = None
        self.TRIAL_LIMIT = None
        self.verbose = False

    def set_verbose(self, to_verbose: bool):
        self.verbose = to_verbose

    def fit(self, population: list[dict], population_limit: dict, max_iter: int = 1000, trial_limit = 10, min_copy = 0, max_copy = 1):
        if len(population) < 1:
            pass

        models = [copy.deepcopy(self.models[0]) for _ in range(len(population))] # len(population) = SN
        self.models = models
        self.max_copy = max_copy
        self.min_copy = min_copy
        self.max_iter = max_iter
        self.POPULATION_LIMIT = population_limit # aka param_ranges
        self.POPULATION_TEMPLATE = population[0]
        self.PARAM_DIMS = get_dims(population[0])
        self.TRIAL_LIMIT = trial_limit

        rng = np.random.default_rng(self.random_state)

        solutions = np.concatenate(tuple(map(lambda x: x.reshape(1,-1), map(as_solution, population)))) # of shape (SN, D)
        scores = [self.compute_fitness(population,i) for i,population in enumerate(population)]
        # scores = list(map(self.compute_fitness, population))
        SN, D = solutions.shape
        indexes = np.arange(SN)

        total_time = 0.
        trials = [0] * SN
        for current_iter in range(max_iter+1):
            # only use current_iter of 1 to max_iter (inclusive)
            if current_iter == 0: continue
            # if self.best_score >= 1: break

            # Timing purpose
            start_time = time.time()

            # Employed bee process, single loop to check whether generated solution are better
            for i in indexes:
                second_solution = rng.choice(solutions[indexes != i])
                v: dict = self.neighbourhood_gen(solutions[i], second_solution, current_iter)
                # Skip if Si is same as Vi
                v_solution = as_solution(v)
                if np.all(v_solution == solutions[i]):
                    trials[i] += 1
                    # print(f'\t{i} Employed: Literally the same')
                    continue
                v_score = self.compute_fitness(v,i)
                # replace Si if Vi is better
                if v_score > scores[i]:
                    solutions[i] = v_solution
                    scores[i] = v_score
                    trials[i] = 0
                else:
                    trials[i] += 1
            # Timing purpose
            end_employed_time = time.time()
            # if self.verbose: print(f"\tEmployed: {end_employed_time - start_time:.4f} seconds")

            fitness_sum = np.sum(scores)
            selection_probability = np.divide(scores, fitness_sum)
            # Onlooker bee process
            second_solutions = [rng.choice(solutions[indexes != i]) for i in indexes]
            for i in indexes:
                solution_idx = rng.choice(indexes, p=selection_probability)
                # solution = solutions[solution_idx]
                # second_solution = rng.choice(solutions[indexes != solution_idx]) # select solutions excluding itself
                v: dict = self.neighbourhood_gen(solutions[i], second_solutions[i], current_iter)
                v_solution = as_solution(v)
                if np.all(v_solution == solutions[i]):
                    trials[i] += 1
                    # print(f'\t{i} Onlooker: Literally the same')
                    continue
                v_score = self.compute_fitness(v,i)
                # Replace Si if Vi is better
                if v_score > scores[solution_idx]:
                    solutions[solution_idx] = v_solution
                    scores[solution_idx] = v_score
                    trials[i] = 0
                else:
                    trials[i] += 1

            # Scout bee process, find new source if score is not improving
            for i in indexes:
                if trials[i] > self.TRIAL_LIMIT-1:
                    # if self.verbose: print(f'\t{i}: Update its solution')
                    solutions[i] = copy.deepcopy(as_solution(
                        Optimiser.population_generation(
                            1, self.POPULATION_TEMPLATE, self.POPULATION_LIMIT, int(rng.uniform(1, 4294967295))
                        )[0]
                    ))
                    trials[i] = 0

            # Save current best solution
            current_best_idx = np.argmax(scores)
            # if self.verbose: print(f"\tCurrent highest score: {scores[current_best_idx]}")

            if scores[current_best_idx] > self.best_score:
                self.best_individual = copy.deepcopy(as_params(solutions[current_best_idx], self.PARAM_DIMS))
                self.best_idx = int(current_best_idx)
                if self.verbose: print(f"\tUpdated score: {scores[current_best_idx]}, idx: {current_best_idx}")
                if self.verbose: print(f'\tBest score: {self.compute_fitness(self.best_individual,self.best_idx)}, idx: {self.best_idx}')
                # self.best_solution = solutions[current_best_idx]
                self.best_score = copy.deepcopy(scores[current_best_idx])

            end_time = time.time()
            total_time += end_time - start_time
            if self.verbose: print(f'Iter {current_iter}/{max_iter}: {end_time - start_time:.4f} seconds   \t Average time: {total_time / current_iter:.4f} seconds')

        if self.verbose: print(f'Best score recorded: {self.best_score}, at idx: {self.best_idx}, total time: {total_time:.4f} seconds')
        # if self.verbose: print(f'Best score: {self.compute_fitness(self.best_individual,self.best_idx)}, idx: {self.best_idx}, ')
        return self.best_individual
    def neighbourhood_gen(self, solution: np.ndarray, solution_k: np.ndarray, curr_iter: int):
        """
        Run with Variable Copy, inspired by spatial distribution in IWO
        :return: dict, not ndarray
        """
        rng = np.random.default_rng(self.random_state)
        p_copy = ABCOptimiser.get_p_copy(curr_iter, self.max_iter, self.min_copy, self.max_copy)
        should_copy = np.less(rng.random(solution.shape), p_copy) # array of booleans
        phis = rng.uniform(-1, 1, solution.shape)
        res = np.where(should_copy, solution, np.add(solution, np.multiply(phis, np.subtract(solution, solution_k))))
        
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
        Variable Copy, inspired by spatial distribution in IWO
        """
        return (max_copy - min_copy) * (curr_iter / max_iter) + min_copy



if __name__ == '__main__':
    from ELM import ELMclf
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv('data_preprocessed.csv', index_col=0)
    booleans = df.drop('Age', axis=1)
    df[booleans.columns] = booleans.astype('int64')

    X = df.drop('Recurred_Yes', axis=1)
    Y = df['Recurred_Yes']
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

    n = 60
    elm = ELMclf(n, random_state=42)
    elm.fit(x_train.to_numpy(), y_train.to_numpy())

    popu_temp = {'weights_i': elm.weights_i, 'biases': elm.biases}
    popu_range = {'weights_i': (0,1), 'biases': (0,1)}
    # popu = Optimiser.population_generation(10, {'biases': elm.biases}, {'biases': (0,1)}, random_state=42)
    popu = Optimiser.population_generation(10, popu_temp, popu_range, random_state=42)

    abc = ABCOptimiser(elm, x_train.to_numpy(), y_train.to_numpy(), x_test.to_numpy(), y_test.to_numpy(), fitness_fn=f1_score, random_state=42)
    opti = GAOptimiser(elm, x_train.to_numpy(), y_train.to_numpy(), x_test.to_numpy(), y_test.to_numpy(), fitness_fn=accuracy_score, random_state=42)
    ori_preci = precision_score(y_test, elm.predict(x_test.to_numpy()))
    print(f'original precision: {ori_preci:.4f}')
    print(f'original f1: {f1_score(y_test, elm.predict(x_test.to_numpy())):.4f}')
    print(confusion_matrix(y_test, elm.predict(x_test.to_numpy())))

    # print(np.concat((as_solution(popu[0]).reshape(1,-1), as_solution(popu[1]).reshape(1,-1))).shape)
    # opti.generation_loop(popu, 10)
    # opti.score()
    abc.set_verbose(True)
    best_param = abc.fit(popu, popu_range, max_iter=1000, trial_limit=100)
    print('Best Params: ', best_param)
    # print(abc.compute_fitness(abc.best_individual,abc.best_idx))
    abc.score()

    # clf = ELMclf(n, random_state=42)
    # clf.fit(x_train.to_numpy(), y_train.to_numpy(), **best_param)
    # test_pred = clf.predict(x_test.to_numpy())
    # print('Precision score: ', precision_score(y_test, test_pred))
    # print(confusion_matrix(y_test, clf.predict(x_test.to_numpy())))
    # opti.score(lambda y,y_pred: precision_score(y, y_pred, average='binary', zero_division=0))

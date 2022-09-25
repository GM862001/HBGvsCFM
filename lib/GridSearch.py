from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from itertools import product
from copy import deepcopy
import numpy as np
import heapq

from .metrics import Metric, metrics_aliases


class GridSearch():

    """
    Grid Search Class.

    Attributes
    ----------
    model (Model): the model whose fitting parameters are to be optimize.
    n_best_params_combinations (int, default: 1): the number of best parameters combination to consider.
    _best_params_combination (list): the list of the best parameters combinations.
    metric (str or Metric, default: "loss"): the alias of the metric to consider the metric itself, or "loss".
    target_dataset ("train" or "test", default: "test"): whether the metric considered must be measured on the training or on the test set.
    mode ("min" or "max", default: "min"): whether the metric considered is to be minimized or maximized.

    Methods
    -------
    __init__(self, model, n_best_params_combinations = 1, metric = "loss", target_dataset = "test", mode = "min"):
        Initializes the parameters of the grid search.
    fit(self, X, y, params_grid, X_test = None, y_test = None, kfold = False, skfold = False, test_split_ratio = 0.1, verbose = True, random_seed = 0):
        Search for the best fitting parameters in the given grid space.

    """

    def __init__(self, model, n_best_params_combinations = 1, metric = "loss", target_dataset = "test", mode = "min"):

        """
        Initializes the parameters of the grid search.

        Parameters
        ----------
        model (Model): the model whose fitting parameters are to be tuned.
        n_best_params_combinations (int, default: 1): the number of best parameters combination to consider.
        metric (str or Metric, default: "loss"): the alias of the metric to consider the metric itself, or "loss".
        target_dataset ("train" or "test", default: "test"): whether the metric considered must be measured on the training or on the test set.
        mode ("min" or "max", default: "min"): whether the metric considered is to be minimized or maximized.

        """

        self.model = model
        self.n_best_params_combinations = n_best_params_combinations
        
        if type(metric) == str:
            if metric in metrics_aliases:
                self.metric = metrics_aliases[metric]
            elif metric == "loss":
                self.metric = metric
            else:
                raise ValueError("Unknown metric.")
        elif issubclass(type(metric), Metric):
            self.metric = metric
        else:
            raise ValueError("The metric must be a Metric object or as string alias for it, or 'loss'.")
        
        if target_dataset not in ["train", "test"]:
            raise ValueError("The target dataset must be either 'train' or 'test'.")
        self.target_dataset = target_dataset
        
        self.mode = mode
    
    def fit(self, X, y, params_grid, X_test = None, y_test = None, kfold = False, skfold = False, test_split_ratio = 0.1, n_random_runs = 1, verbose = True, random_seed = 0):

        """
        Search for the best fitting parameters in the given grid space.

        Parameters
        ----------
        X (np.ndarray, shape: (training_set_size, n_input_features)): the training set input features.
        y (np.ndarray, shape: (training_set_size, n_output_features)): the training set ground truth.
        params_grid (dict): a dictionary whose keys are the fitting parameters to tune and whose values are the list of values to consider for each of them.
        X_test (np.ndarray, shape: (test_set_size, n_input_features)): the test set input features. Only used when the target dataset is the test one.
        y_test (np.ndarray, shape: (test_set_size, n_output_features)): the test set ground trugh. Only used when the target dataset is the test one.
        kfold (bool, default: False): whether to apply or not k-fold cross validation. Must only be used when the target dataset is the test one and no test set is provided.
        skfold (bool, default: False): whether to apply or not stratified k-fold cross validation. Must only be used when the target dataset is the test one and no test set is provided.
        test_split_ratio (float, default: 0.1): the ratio between the test set size and the training set one when the target dataset is the test one and no test set is provided.
        n_random_runs (int, default: 1): the number or random runs to execute (each with a different weights initialization) for each parameter combination.
        verbose (bool, default: True): whether to activate or not verbose mode.
        random_seed (int, default: 0): random seed for test dataset drawning.

        """

        if self.target_dataset == "test" and not X_test:
            if kfold:
                folds = KFold(n_splits = int(1 / test_split_ratio), shuffle = True, random_state = random_seed)
            elif skfold:
                folds = StratifiedKFold(n_splits = int(1 / test_split_ratio), shuffle = True, random_state = random_seed)
            else:
                X, X_test, y, y_test = train_test_split(X, y, test_size = test_split_ratio, shuffle = True, random_state = random_seed)

        params_combinations = list(product(*list(params_grid.values())))
        results = []

        for i, params_combination in enumerate(params_combinations):

            temp_result = {}
            temp_params = {}
            for j, key in enumerate(params_grid.keys()):
                temp_params[key] = params_combination[j]

            if verbose:
                print(f"*** Parameters combination {i + 1}/{len(params_combinations)} ***\n")
                print(temp_params)

            if kfold or skfold:
                folds_scores = []
                for j, (train, test) in enumerate(folds.split(X, y)):
                    if verbose:
                        print(f"\nFold {j + 1}")
                    runs_scores = []
                    for run in range(n_random_runs):
                        if verbose:
                            print(f"\nRun {run + 1}")
                        self.model.fit(X[train], y[train], **temp_params, random_seed = run)
                        run_score = self.model.score(X[test], y[test], self.metric)
                        runs_scores.append(run_score)
                        if verbose:
                            print("Run score: ", run_score)
                    fold_score = np.mean(runs_scores)
                    folds_scores.append(fold_score)
                    if verbose:
                        print("Fold score: ", fold_score)
                temp_result["score"] = np.mean(folds_scores)
                temp_result["score_std"] =  np.std(folds_scores)
            else:
                runs_scores = []
                for run in range(n_random_runs):
                    if verbose:
                        print(f"\nRun {run + 1}")
                    self.model.fit(X, y, **temp_params, random_seed = run)
                    if self.target_dataset == "test":
                        run_score = self.model.score(X_test, y_test, self.metric)
                    else:
                        run_score = self.model.score(X, y, self.metric)
                    runs_scores.append(run_score)
                    if verbose:
                        print("Run score: ", run_score)
                temp_result["score"] = np.mean(runs_scores)
                temp_result["score_std"] =  np.std(runs_scores)

            if verbose:
                print(f"\nScore: {temp_result['score']}\n")

            temp_result["params"] = deepcopy(temp_params)
            results.append(temp_result)

        if self.mode == "min":
            self._best_params_combinations = heapq.nsmallest(self.n_best_params_combinations, results, key = lambda tr: tr["score"])
        else:
            self._best_params_combinations = heapq.nlargest(self.n_best_params_combinations, results, key = lambda tr: -tr["score"])

        if verbose:
            print("*** Best parameters combinations ***")
            for i, best_params_combination in enumerate(self._best_params_combinations, start = 1):
                print(f"N. {i}")
                print(best_params_combination["params"])
                print("Score: ", best_params_combination["score"])
from sklearn.model_selection import train_test_split
import numpy as np
from copy import deepcopy

from .metrics import Metric, ErrorFunction, metrics_aliases, error_functions_aliases
from .regularization_functions import RegularizationFunction
from .regularization_functions import aliases as regularization_functions_aliases
from .optimizers import Optimizer
from .optimizers import aliases as optimizers_aliases
from .layers import DenseLayer, FullyConnectedLayer


class Model():

    """
    Model Base Class.

    Base methods
    ------------
    __call__(self, X) -> np.ndarray:
        Returns the predictions of the model for the given input.
    score(self, X, y, metric = "loss") -> dict:
        Computes the score of the model for the given input according to the given metric.

    Methods to override
    -------------------
    predict(self, X) -> np.ndarray:
        Returns the predictions of the model for the given input.
    compute_loss(self, y_true, y_pred) -> float:
        Computes the loss of the model for the given set of predictions and ground truth.
    fit(self, X, y, random_seed = 0):
        Fits the parameters of the model to the given data.

    """

    def predict(self, X):

        """
        Returns the predictions of the model for the given input.

        Parameters
        ----------
        X (np.ndarray, shape: (n_samples, n_input_features)): the input of the model.

        Returns (np.ndarray, shape: (n_samples, n_output_features)): the predictions of the model for the given input.
        
        """

        raise NotImplementedError

    def __call__(self, X):

        """
        Returns the predictions of the model for the given input.

        Parameters
        ----------
        X (np.ndarray, shape: (n_samples, n_input_features)): the input of the model.

        Returns (np.ndarray, shape: (n_samples, n_output_features)): the predictions of the model for the given input.
        
        """

        return self.predict(X)

    def compute_loss(self, y_true, y_pred):

        """
        Computes the loss of the model for the given set of predictions and ground truth.

        Parameters
        ----------
        y_true (np.ndarray, shape: (n_samples, n_output_features)): the ground truth values of the target variable.
        y_pred (np.ndarray, shape: (n_samples, n_output_features)): the predicted values of the target variable.

        Returns (float): the loss of the model for the given set of predictions and ground truth.

        """

        raise NotImplementedError

    def score(self, X, y, metric = "loss"):

        """
        Computes the score of the model for the given input according to the given metric.
        
        Parameters
        ----------
        X (np.ndarray, shape: (n_samples, n_input_features)): the input of the model.
        y (np.ndarray, shape: (n_samples, n_output_features)): the ground truth values for the target variable. 
        metrics (str or Metric, default: "loss"): the metric to evaluate, an alias for it, or "loss".

        Returns (dict): a dictionary whose keys are the metrics considered and whose values are the realtive scores.

        """

        y_pred = self(X)

        if type(metric) == str:
            if metric in metrics_aliases:
                return metrics_aliases[metric](y, y_pred)
            elif metric == "loss":
                return self.compute_loss(y, y_pred)
            else:
                raise ValueError("Unknown metric")
        elif issubclass(type(metric), Metric):
            return metric(y, y_pred)
        else:
            raise ValueError("The metric must be a Metric object or a string aliases for it, or 'loss'.")

    def fit(self, X, y, random_seed = 0):

        """
        Fits the parameters of the model to the given data.

        Parameters
        ----------
        X (np.ndarray, shape: (training_set_size, n_input_features)): the training set input features.
        y (np.ndarray, shape: (training_set_size, n_output_features)): the training set ground truth.
        random_seed (int, default: 0): random seed.

        """

        raise NotImplementedError

class Sequential(Model):

    """
    Sequential Model Class.

    Attributes
    ----------
    _layers (list<Layer>): the layers of the model.
    _eval (dict): dictionary with fitting evaluation purposes.
    _error_function (ErrorFunction, default: MSE()): the error function to consider when training the model.
    _regularization_function (RegularizationFunction, default: NullRegularization()): the regularization function to consider when training the model.

    Methods
    -------
    __init__(self):
        Initializes the layers list of the model.
    add(self, layer):
        Attaches a layer to the model.
    compute_loss(self, y_true, y_pred) -> float:
        Computes the loss of the network for the given set of predictions and ground truth.
    _forward(self, X) -> np.ndarray, shape: (batch_size, n_output_features):
        Forward propagates through the model.
    predict(self, X) -> np.ndarray, shape: (batch_size, n_output_features):
        Returns the predictions of the model for a given input.
    _backward(self, output, y):
        Backward propagates through the model.
    _learn(self, X, y):
        Forward and backward propagates through the model.
    compile(self, error_function = "MSE", regularization_function = "Null"):
        Initializes the loss function of the model before the start of training.
    fit(self, X, y, X_test = None, y_test = None, optimizer = "HBG", epochs = 100, minibatch_size = None, early_stopping = None, test_split_ratio = 0.1, metrics = [], verbose = True, random_seed = 0):
        Fits the parameters of the model to the given data.

    """

    def __init__(self):

        """
        Initializes the layers list of the model.

        """

        self._layers = []

    def add(self, layer):

        """
        Attaches a layer to the model.
        
        Parameters
        ----------
        layer (Layer): the layer to attach.

        """

        self._layers.append(layer)

    def compute_loss(self, y_true, y_pred):

        """
        Computes the loss of the model for the given set of predictions and ground truth.

        Parameters
        ----------
        y_true (np.ndarray, shape: (n_samples, n_output_features)): the ground truth values of the target variable.
        y_pred (np.ndarray, shape: (n_samples, n_output_features)): the predicted values of the target variable.

        Returns (float): the loss of the model for the given set of predictions and ground truth.

        """
        regularization_term = 0
        for layer in self._layers:
            regularization_term += layer.compute_regularization(self._regularization_function)

        return self._error_function(y_true, y_pred) + regularization_term

    def _forward(self, X):

        """
        Forward propagates through the model.

        Parameters
        ----------
        X (np.ndarray, shape: (batch_size, n_input_features)): the input of the model.

        Returns (np.ndarray, shape: (batch_size, n_output_features)): the output of the model.
        
        """
        
        output = X
        for layer in self._layers:
            output = layer.forward(output)
        return output

    def predict(self, X):

        """
        Returns the predictions of the model for the given input.

        Parameters
        ----------
        X (np.ndarray, shape: (n_samples, n_input_features)): the input of the model.

        Returns (np.ndarray, shape: (n_samples, n_output_features)): the predictions of the model for the given input.
        
        """

        return self._forward(X)

    def _backward(self, output, y):

        """
        Backward propagates through the model.
        
        Parameters
        ----------
        output (np.ndarray, shape: (batch_size, n_output_features)): the output of the model for a given forward pass. 
        y (np.ndarray, shape: (batch_size, n_output_features)): the ground truth values of the target variable.

        """

        grad_output = self._error_function.derivative(y, output)

        for layer in reversed(self._layers):
            grad_output = layer.backward(grad_output, self._regularization_function)

    def _learn(self, X, y):

        """
        Forward and backward propagates through the model.

        Parameters
        ----------
        X (np.ndarray, shape: (batch_size, n_input_features)): the input of the model.
        y (np.ndarray, shape: (batch_size, n_output_features)): the ground truth values of the target variable.

        """

        output = self._forward(X)

        for layer in self._layers:
            layer.after_forward_propagation(self.compute_loss(y, output))

        self._backward(output, y)

    def compile(self, error_function = "MSE", regularization_function = "Null"):

        """
        Initializes the loss function of the model before the start of training.

        Parameters
        ----------
        error_function (str or ErrorFunction, default: "MSE"): an alias for the error function to consider when training the model or the error function itself.
        regularization_function (str or RegularizationFunction, default: "Null"): an alias for the regularization function to consider when training the model or the regularization function itself.

        """

        if type(error_function) == str:
            if error_function not in error_functions_aliases:
                raise ValueError("Unknown error function")
            self._error_function = error_functions_aliases[error_function]
        elif issubclass(type(error_function), ErrorFunction):
            self._error_function = error_function
        else:
            raise ValueError("The error function must be an ErrorFunction object or a string alias for it.")

        if type(regularization_function) == str:
            if regularization_function not in regularization_functions_aliases:
                raise ValueError("Unknown error function")
            self._regularization_function = regularization_functions_aliases[regularization_function]
        elif issubclass(type(regularization_function), RegularizationFunction):
            self._regularization_function = regularization_function
        else:
            raise ValueError("The regularization function must be a RegularizationFunction object or a string alias for it.")

    def fit(self, X, y, X_test = None, y_test = None, optimizer = "HBG", epochs = 100, minibatch_size = None, early_stopping = None, test_split_ratio = 0.1, metrics = [], verbose = True, random_seed = 0):

        """
        Fits the parameters of the model to the given data.

        Parameters
        ----------
        X (np.ndarray, shape: (training_set_size, n_input_features)): the training set input features.
        y (np.ndarray, shape: (training_set_size, n_output_features)): the training set ground trugh.
        X_test (np.ndarray, shape: (test_set_size, n_input_features), default: None): the test set input features.
        y_test (np.ndarray, shape: (test_set_size, n_output_features), default: None): the test set ground truth.
        optimizer (str or Optimizer, default: "HBG"): an alias for the optimizer to use for the fitting of the model parameters or the optimizer itself..
        epochs (int, default: 100): the number of epochs of training.
        minibatch_size (int, default: None): if not None, minibatch training is performed with minibatches of this size.
        early_stopping (EarlyStopping, default: None): early stopping.
        test_split_ratio (float, default: 0.1): if early stopping is applied to a metric measured on the test set, and no test set is provided, then one is drawn from the training set according to this splitting ratio.
        metrics (list<str or Metric>, default: []): the list of the metrics to evaluate during the training or aliases for them.
        verbose (bool, default: True): whether to activate or not verbose mode.
        random_seed (int, default: 0): random seed.

        """

        # Dereferencing the optimizer alias
        if type(optimizer) == str:
            if optimizer not in optimizers_aliases:
                raise ValueError("Unknown optimizer")
            else:
                optimizer = optimizers_aliases[optimizer]
        elif issubclass(type(optimizer), Optimizer):
            optimizer = optimizer
        else:
            raise ValueError("The optimizer must be an Optimizer object or a string alias for it.")

        # Dereferencing the metrics aliases
        for i, metric in enumerate(metrics):
            if type(metric) == str:
                if metric not in metrics_aliases:
                    raise ValueError("Unknown metric")
                metrics[i] = metrics_aliases[metric]
            elif not issubclass(type(metric), Metric):
                raise ValueError("Metrics must be Metric objects or string aliases for them.")

        # Initialization of the parameters of the model before the start of training.
        np.random.seed(random_seed)
        for layer in self._layers:
            layer.initialize(deepcopy(optimizer))
        if early_stopping:
            early_stopping.initialize()
            if early_stopping.target_dataset == "test" and not X_test:
                X, X_test, y, y_test = train_test_split(X, y, test_size = test_split_ratio, shuffle = True, random_state = random_seed)

        # self._eval set up
        self._eval = {"train_loss": []}
        for metric in metrics:
            self._eval[f"train_{metric}"] = []
        if  X_test is not None:
            self._eval["test_loss"] = []
            for metric in metrics:
                self._eval[f"test_{metric}"] = []

        # Training

        for epoch in range(1, epochs + 1):

            if verbose:
                print(f"EPOCH {epoch} -> ", end = '')

            # Learning
            if minibatch_size is not None:
                indices = np.arange(X.shape[0])
                np.random.shuffle(indices)
                for start_idx in range(0, indices.shape[0] - minibatch_size + 1, minibatch_size):
                    batch_idx = indices[start_idx : start_idx + minibatch_size]
                    self._learn(X[batch_idx], y[batch_idx])
            else:
                self._learn(X, y)

            # Training statistics recording
            output = self._forward(X)
            loss = self.compute_loss(y, output)
            self._eval["train_loss"].append(loss)
            if verbose:
                print(f"Train loss: {loss}", end = ' ')
            for metric in metrics:
                metric_value = metric(y, output)
                self._eval[f"train_{metric}"].append(metric_value)
                if verbose:
                    print(f"Train {metric}: {metric_value}", end = ' ')

            # Test statistics recording
            if X_test is not None:
                test_output = self._forward(X_test)
                test_loss = self.compute_loss(y_test, test_output)
                self._eval["test_loss"].append(test_loss)
                if verbose:
                    print(f"Test loss: {test_loss}", end = ' ')
                for metric in metrics:
                    metric_value = metric(y_test, test_output)
                    self._eval[f"test_{metric}"].append(metric_value)
                    if verbose:
                        print(f"Test {metric}: {metric_value}", end = ' ')

            for layer in self._layers:
                layer.on_epoch_end()

            # Early stopping
            if early_stopping:
                params = [layer.get_params() for layer in self._layers]
                if early_stopping.target_dataset == "train":
                    stop = early_stopping.on_epoch_end(loss, y, output, params)
                else:
                    stop = early_stopping.on_epoch_end(test_loss, y_test, test_output, params)
                if stop:
                    if verbose:
                        print("\nEarly stopped training")
                    best_params = early_stopping._best_params
                    for layer, layer_best_params in zip(self._layers, best_params):
                        layer.set_params(layer_best_params)
                    break

            if verbose:
                print()


class MLP(Sequential):

    """
    Multi Layer Perceptron Class.

    Methods
    -------
    __init___(self, layer_sizes, activation = "relu", weights_initialization = "scaled", weights_scale = 0.01):
        Initializes the layers of the MLP.

    """

    def __init__(self, layer_sizes, activation = "relu", weights_initialization = "scaled", weights_scale = 0.01):

        """
        Initializes the layers of the MLP.

        Parameters
        ----------
        layer_sizes (list<int>): the sizes of the layers of the MLP.
        activation (str or ActivationFunction): alias for the activation function of the MLP or the activation function itself.
        weights_initialization (str, default: "scaled"): the weights initialization strategy of the layers of the network. Must be either "xavier", "he" or "scaled".
        weights_scale (float, default: 0.01): the scale of the weights initialization. Used only when the weights strategy initialization is "scaled".

        """

        super().__init__()

        # Hidden layers
        for n_input, n_output in zip(layer_sizes[:-1], layer_sizes[1:-1]):
            layer = DenseLayer(n_input, n_output, activation, weights_initialization, weights_scale)
            self.add(layer)
        
        # Output layer
        n_input, n_output = layer_sizes[-2:]
        layer = FullyConnectedLayer(n_input, n_output, weights_initialization, weights_scale)
        self.add(layer)

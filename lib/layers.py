import numpy as np

from .activation_functions import ActivationFunction
from .activation_functions import aliases as activation_functions_aliases


class Layer():

    """
    Layer Base Class.

    Base Methods
    ------------
    get_params(self) -> dict:
        Returns the parameters of the layer in the form of a dictionary.
    set_params(self, params):
        Sets the parameters of the layer to the given values, if any.
    initialize(self, optimizer):
        Initializes the layer before the start of training, if needed.
	after_forward_propagation(self, loss):
		Updates the parameters of the layer at the end of each forward pass, if needed.
    on_epoch_end(self):
        Updates the parameters of the layer at the end of each epoch of training, if needed.
    compute_regularization(self, regularization_function) -> float:
        Computes the value of the regularization function term of the loss function due to the parameters of the layer.

    Methods to override
    -------------------
    forward(self, input) -> np.ndarray:
        Forward propagates through the layer.
    backward(self, grad_output, regularization_function) -> np.ndarray:
        Backward propagates through the layer.

    """

    def get_params(self):

        """
        Returns the parameters of the layer in the form of a dictionary.

        """

        return dict()

    def set_params(self, params):

        """
        Sets the parameters of the layer to the given values, if any.

        Parameters
        ----------
        params (dict): the values to set for the parameters of the layer.

        """

        pass

    def initialize(self, optimizer):

        """
        Initializes the layer before the start of training, if needed.

        Parameters
        ----------
        optimizer (Optimizer): the optimizer to use for the fitting of the layer parameters.

        """

        pass

    def forward(self, input):

        """
        Forward propagates through the layer.

        Parameters
        ----------
        input (np.ndarray, shape: (batch_size, n_input_features)): the input of the layer.

        Returns (np.ndarray, shape: (batch_size, n_output_features)): the output of the layer.

        """

        raise NotImplementedError

    def compute_regularization(self, regularization_function):

        """
        Computes the value of the regularization term of the loss function due to the parameters of the layer.

        Parameters
        ----------
        regularization_function (RegularizationFunction): the regularization function considered.

        Returns (float): the value of the regularization term of the loss function due to the parameters of the layer.

        """

        return 0

    def after_forward_propagation(self, loss):

        """
        Updates the parameters of the layer at the end of each forward pass, if needed.

        Parameters
        ----------
        loss (float): the value of the loss at the current epoch of training.

        """

        pass

    def on_epoch_end(self, loss):

        """
        Updates the parameters of the layer at the end of each epoch of training, if needed.

        Parameters
        ----------
		loss (float): the value of the loss at the current epoch of training.

        """

        pass

    def backward(self, grad_output, regularization_function):

        """
        Backward propagates through the layer.
        
        Parameters
        ----------
        grad_output (np.ndarray, shape: (batch_size, n_output_features)): the derivative of the error function term of the loss function with respect to the output of the layer.
        regularization_function (RegularizationFunction): the regularization function considered.

        Returns (np.ndarray, shape: (batch_size, n_input_features)): the derivative of the error function term of the loss function with respect to the input of the layer.

        """

        raise NotImplementedError


class FullyConnectedLayer(Layer):

    """
    Fully Connected Layer Class.
    A layer which outputs a linear combination of its inputs.

    Attributes
    ----------
    n_input (int): the number of input units of the layer.
    n_output (int): the number of output units of the layer.
    weights_initialization (str, default: "scaled"): the weights initialization strategy of the layer. Must be either "xavier", "he" or "scaled".
    weights_scale (float, default: 0.01): the scale of the weights initialization. Used only when the weights strategy initialization is "scaled".
    _weights (np.ndarray, shape: (n_input, n_output)): the weights of the layer.
    _biases (np.ndarray, shape: (1, n_output)): the biases of the layer.
    _input (np.ndarray, shape: (batch_size, n_input)): the input of the layer for a given forward-backward pass.
    optimizer (Optimizer): the optimizer to use for the fitting of the layer parameters.

    Methods
    -------
    __init__(self, n_input, n_output, weights_initialization = "scaled", weights_scale = 0.01):
        Initializes the topology and the weights initialization strategy of the layer.
    get_params(self) -> dict:
        Returns the parameters of the layer in the form of a dictionary.
    set_params(self, params):
        Sets the parameters of the layer to the given values.
    initialize(self, optimizer):
        Initializes the layer before the start of training.
    forward(self, input) -> np.ndarray:
        Forward propagates through the layer.
    compute_regularization(self, regularization_function) -> float:
        Computes the value of the regularization term of the loss function due to the parameters of the layer.
	after_forward_propagation(self, loss):
		Updates the parameters of the layer at the end of each forward pass.
    on_epoch_end(self, loss):
        Updates the parameters of the layer at the end of each epoch of training.
    backward(self, grad_output, loss_function):
        Backward propagates through the layer.

    """

    def __init__(self, n_input, n_output, weights_initialization = "scaled", weights_scale = 0.01):

        """
        Initializes the topology and the weights initialization strategy of the layer.

        Parameters
        ----------
        n_input (int): the number of input units of the layer.
        n_output (int): the number of output units of the layer.
        weights_initialization (str, default: "scaled"): the weights initialization strategy of the layer. Must be either "xavier", "he" or "scaled".
        weights_scale (float, default: 0.01): the scale of the weights initialization. Used only when the weights strategy initialization is "scaled".

        """

        self.n_input = n_input
        self.n_output = n_output

        if weights_initialization not in ["xavier", "he", "scaled"]:
            raise ValueError("Weights initialization must be 'xavier', 'he' or 'scaled'.")
        self.weights_initialization = weights_initialization
        if weights_initialization == "scaled":
            self.weights_scale = weights_scale

    def get_params(self):

        """
        Returns the parameters of the layer in the form of a dictionary.

        Returns (dict): the parameters of the layer.

        """

        return dict(weights = self._weights.copy(), biases = self._biases.copy())

    def set_params(self, params):

        """
        Sets the parameters of the layer to the given values.

        Parameters
        ----------
        params (dict): the values to set for the parameters of the layer.

        """

        self._weights = params["weights"]
        self._biases = params["biases"]

    def initialize(self, optimizer):

        """
        Initializes the layer before the start of training.

        Parameters
        ----------
        optimizer (Optimizer): the optimizer to use for the fitting of the layer parameters.

        """

        if self.weights_initialization == "xavier":
            scale = 1 / self.n_input
        elif self.weights_initialization == "he":
            scale = 2 / self.n_input
        else: # self.weights_initialization == "scaled"
            scale = self.weights_scale
        self._weights = np.random.normal(loc = 0.0, scale = scale, size = (self.n_input, self.n_output))
        self._biases = np.zeros((1, self.n_output))
        self.optimizer = optimizer
        self.optimizer.initialize()

    def forward(self, input):

        """
        Forward propagates through the layer.

        Parameters
        ----------
        input (np.ndarray, shape: (batch_size, n_input)): the input of the layer.

        Returns (np.ndarray, shape: (batch_size, n_output)): the output of the layer.

        """

        self._input = input
        return np.dot(self._input, self._weights) + self._biases

    def after_forward_propagation(self, loss):

        """
        Updates the parameters of the layer at the end of each forward pass.

        Parameters
        ----------
		loss (float): the value of the loss at the current epoch of training.

        """

        self.optimizer.after_forward_propagation(loss)

    def on_epoch_end(self):

        """
        Updates the parameters of the layer at the end of each epoch of training.

        """

        self.optimizer.on_epoch_end()

    def compute_regularization(self, regularization_function):

        """
        Computes the value of the regularization term of the loss function due to the parameters of the layer.

        Parameters
        ----------
        regularization_function (RegularizationFunction): the regularization function considered.

        Returns (float): the regularization term of the loss function due to the parameters of the layer.

        """

        return regularization_function(self._weights)

    def backward(self, grad_output, regularization_function):

        """
        Backward propagates through the layer.

        Parameters
        ----------
        grad_output (np.ndarray, shape: (batch_size, n_output)): the derivative of the error function with respect to the output of the layer.
        regularization_function (RegularizationFunction): the regularization function considered.

        Returns (np.ndarray, shape: (batch_size, n_input)): the derivative of the error function with respect to the inputs of the layer.

        """

        grad_input = np.dot(grad_output, self._weights.T) # (np.ndarray, shape: (batch_size, n_input)): derivative of the error function with respect to the inputs of the layer
        grad_weights = np.dot(self._input.T, grad_output) + regularization_function.derivative(self._weights) # (np.ndarray, shape: (n_input, n_output)): derivative of the loss function with respect to the weights of the layer
        grad_biases = grad_output.sum(axis = 0, keepdims = True) # (np.ndarray, shape: (1, n_output)): derivative of the loss function with respect to the biases of the layer
        grad_params = np.concatenate((grad_biases, grad_weights))
        params_update = self.optimizer.optimize(grad_params)
        biases_update, weights_update = np.vsplit(params_update, [1])
        self._biases += biases_update
        self._weights += weights_update
        return grad_input


class ActivationLayer(Layer):

    """
    Activation Layer Class.
    A layer which outputs an activation function of its inputs.

    Attributes
    ----------
    activation (ActivationFunction, default: ReLU()): the activation function of the layer.
    _input (np.ndarray, shape: (batch_size, n_units)): the input of the layer for a given forward-backward pass.

    Methods
    -------
    __init__(self, activation = "relu"):
        Initializes the activation function of the layer.
    forward(self, input) -> np.ndarray:
        Forward propagates through the layer.
    backward(self, grad_output, regularization_function) -> np.ndarray:
        Backward propagates through the layer.

    """

    def __init__(self, activation = "relu"):

        """
        Initializes the activation function of the layer.

        Parameters
        ----------
        activation (str or ActivationFunction, default: "relu"): alias for the activation function of the layer or the activation function itself.

        """

        if type(activation) == str:
            if activation not in activation_functions_aliases:
                raise ValueError("Unknown activation function")
            self.activation = activation_functions_aliases[activation]
        elif issubclass(type(activation), ActivationFunction):
            self.activation = activation
        else:
            raise ValueError("The activation of the layer must be an ActivationFunction object or a string alias for it.")

    def forward(self, input):

        """
        Forward propagates through the layer.

        Parameters
        ----------
        input (np.ndarray, shape: (batch_size, n_units)): the input of the layer.

        Returns (np.ndarray, shape: (batch_size, n_units)): the output of the layer.

        """

        self._input = input
        return self.activation(self._input)

    def backward(self, grad_output, regularization_function):

        """
        Backward propagates through the layer.
        
        Parameters
        ----------
        grad_output (np.ndarray, shape: (batch_size, n_units)): the derivative of the error function with respect to the output of the layer.
        regularization_function (RegularizationFunction): the regularization function considered.

        Returns (np.ndarray, shape: (batch_size, nunits)): the derivative of the error function with respect to the input of the layer.

        """

        return grad_output * self.activation.derivative(self._input)


class DenseLayer(Layer):

    """
    Dense layer class.
    A layer which outputs the activation function of a linear combination of its inputs, being the concatenation of a fully connected layer and an activation layer.

    Attributes
    ----------
    _fully_connected_layer (FullyConnectedLayer): the fully connected part of the layer.
    _activation_layer (ActivationLayer): the activation part of the layer.

    Methods
    -------
    __init__(self, n_input, n_output, weights_initialization = "scaled", weights_scale = 0.01):
        Initializes the fully connected and activation parts of the layer.
    get_params(self) -> dict:
        Returns the parameters of the layer in the form of a dictionary.
    set_params(self, params):
        Sets the parameters of the layer to the given values.
    initialize(self, optimizer):
        Initializes the layer before the start of training.
    forward(self, input) -> np.ndarray:
        Forward propagates through the layer.
    compute_regularization(self, regularization_function) -> float:
        Computes the value of the regularization term of the loss function due to the parameters of the layer.
	after_forward_propagation(self, loss):
		Updates the parameters of the layer at the end of each forward pass.
    on_epoch_end(self, loss):
        Updates the parameters of the layer at the end of each epoch of training.
    backward(self, grad_output, regularization_function) -> np.ndarray:
        Backward propagates through the layer.


    """

    def __init__(self, n_input, n_output, activation = "relu", weights_initialization = "scaled", weights_scale = 0.01):

        """
        Initializes the fully connected and activation parts of the layer.

        Parameters
        ----------
        n_input (int): the number of input units of the layer.
        n_output (int): the number of output units of the layer.
        weights_initialization (str, default: "scaled"): the weights initialization strategy of the layer. Must be either "xavier", "he" or "scaled".
        weights_scale (float, default: 0.01): the scale of the weights initialization. Used only when the weights strategy initialization is "scaled".
        activation (str or ActivationFunction, default: "relu"): alias for the activation function of the layer or the activation function itself.

        """

        self._fully_connected_layer = FullyConnectedLayer(n_input = n_input, n_output = n_output, weights_initialization = weights_initialization, weights_scale = weights_scale)
        self._activation_layer = ActivationLayer(activation = activation)

    def get_params(self):

        """
        Returns the parameters of the layer in the form of a dictionary.

        Returns (dict): the parameters of the layer.

        """

        return self._fully_connected_layer.get_params()

    def set_params(self, params):

        """
        Sets the parameters of the layer to the given values.

        Parameters
        ----------
        params (dict): the values to set for the parameters of the layer.

        """

        self._fully_connected_layer.set_params(params)

    def initialize(self, optimizer):
        
        """
        Initializes the layer before the start of training.

        Parameters
        ----------
        optimizer (Optimizer): the optimizer to use for the fitting of the layer parameters.

        """

        self._fully_connected_layer.initialize(optimizer)

    def forward(self, input):

        """
        Forward propagates through the layer.

        Parameters
        ----------
        input (np.ndarray, shape: (batch_size, n_input)): the input of the layer.

        Returns (np.ndarray, shape: (batch_size, n_output)): the output of the layer.

        """

        net = self._fully_connected_layer.forward(input)
        return self._activation_layer.forward(net)

    def after_forward_propagation(self, loss):

        """
        Updates the parameters of the layer at the end of each forward pass.

        Parameters
        ----------
		loss (float): the value of the loss at the current epoch of training.

        """

        self._fully_connected_layer.after_forward_propagation(loss)

    def on_epoch_end(self):

        """
        Updates the parameters of the layer at the end of each epoch of training.

        """

        self._fully_connected_layer.on_epoch_end()

    def compute_regularization(self, regularization_function):

        """
        Computes the regularization term of the loss due to the parameters of the layer.
        
        Parameters
        ----------
        regularization_function (RegularizationFunction): the regularization function considered.

        Returns (float): the regularization term of the loss due to the parameters of the layer.

        """

        return self._fully_connected_layer.compute_regularization(regularization_function)

    def backward(self, grad_output, regularization_function):

        """
        Backward propagates through the layer.
        
        Parameters
        ----------
        grad_output (np.ndarray, shape: (batch_size, n_output)): the derivative of the error function with respect to the output of the layer.
        regularization_function (RegularizationFunction): the regularization function considered.

        Returns (np.ndarray, shape: (batch_size, n_units)): the derivative of the error function with respect to the input of the layer.

        """

        grad_output = self._activation_layer.backward(grad_output, regularization_function)
        return self._fully_connected_layer.backward(grad_output, regularization_function)

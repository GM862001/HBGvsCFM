import numpy as np


class ActivationFunction():

    """
    Activation Function Base Class. 

    Methods to override
    -------------------
    __call__(self, x) -> np.ndarray:
        Computes the value of the activation function for the given input.
    derivative(self, x) -> np.ndarray:
        Computes the value of the derivative of the activation function for the given input.

    """

    def __call__(self, x):

        """
        Computes the value of the activation function for the given input.

        Parameters
        ----------
        x (np.ndarray, shape: input_shape): the input of the activation function.

        Returns (np.ndarray, shape: input_shape): the value of the activation function for the given input.
    
        """

        raise NotImplementedError

    def derivative(self, x):

        """
        Computes the value of the derivative of the activation function for the given input.

        Parameters
        ----------
        x (np.ndarray, shape: input_shape): the input of the derivative of the activation function.

        Returns (np.ndarray, shape: input_shape): the value of the derivative of the activation function for the given input.
    
        """

        raise NotImplementedError


class Identity(ActivationFunction):

    """
    Identity Activation Function Class.

    Methods
    -------
    __call__(self, x) -> np.ndarray:
        Computes the value of the identity activation function for the given input.
    derivative(self, x) -> np.ndarray:
        Computes the value of the derivative of the identity activation function for the given input.

    """

    def __call__(self, x):

        """
        Computes the value of the identity activation function for the given input.

        Parameters
        ----------
        x (np.ndarray, shape: input_shape): the input of the identity activation function.

        Returns (np.ndarray, shape: input_shape): the value of the identity activation function for the given input.
    
        """

        return x

    def derivative(self, x):

        """
        Computes the value of the derivative of the identity activation function for the given input.

        Parameters
        ----------
        x (np.ndarray, shape: input_shape): the input of the derivative of the identity activation function.

        Returns (np.ndarray, shape: input_shape): the value of the derivative of the identity activation function for the given input.
    
        """

        return np.ones(x.shape)


def sigmoid(x):

    return 1 / (1 + np.exp(-x))

class Sigmoid(ActivationFunction):

    """
    Sigmoid Activation Function Class.

    Methods
    -------
    __call__(self, x) -> np.ndarray:
        Computes the value of the sigmoid activation function for the given input.
    derivative(self, x) -> np.ndarray:
        Computes the value of the derivative of the sigmoid activation function for the given input.

    """

    def __call__(self, x):

        """
        Computes the value of the sigmoid activation function for the given input.

        Parameters
        ----------
        x (np.ndarray, shape: input_shape): the input of the sigmoid activation function.

        Returns (np.ndarray, shape: input_shape): the value of the sigmoid activation function for the given input.
    
        """

        return sigmoid(x)

    def derivative(self, x):

        """
        Computes the value of the derivative of the sigmoid activation function for the given input.

        Parameters
        ----------
        x (np.ndarray, shape: input_shape): the input of the derivative of the sigmoid activation function.

        Returns (np.ndarray, shape: input_shape): the value of the derivative of the sigmoid activation function for the given input.
    
        """

        return sigmoid(x) * (1 - sigmoid(x))


class ReLU(ActivationFunction):

    """
    Rectified Linear Unit (ReLU) Activation Function Class.

    Methods
    -------
    __call__(self, x) -> np.ndarray:
        Computes the value of the ReLU activation function for the given input.
    derivative(self, x) -> np.ndarray:
        Computes the value of the derivative of the ReLU activation function for the given input.

    """

    def __call__(self, x):

        """
        Computes the value of the ReLU activation function for the given input.

        Parameters
        ----------
        x (np.ndarray, shape: input_shape): the input of the ReLU activation function.

        Returns (np.ndarray, shape: input_shape): the value of the ReLU activation function for the given input.
    
        """

        return np.maximum(0, x)

    def derivative(self, x):

        """
        Computes the value of the derivative of the ReLU activation function for the given input.

        Parameters
        ----------
        x (np.ndarray, shape: input_shape): the input of the derivative of the ReLU activation function.

        Returns (np.ndarray, shape: input_shape): the value of the derivative of the ReLU activation function for the given input.
    
        """

        return x > 0


class Tanh(ActivationFunction):

    """
    Hyperbolic Tangent (tanh) Activation Function Class.

    Methods
    -------
    __call__(self, x) -> np.ndarray:
        Computes the value of the tanh activation function for the given input.
    derivative(self, x) -> np.ndarray:
        Computes the value of the derivative of the tanh activation function for the given input.

    """

    def __call__(self, x):

        """
        Computes the value of the tanh activation function for the given input.

        Parameters
        ----------
        x (np.ndarray, shape: input_shape): the input of the tanh activation function.

        Returns (np.ndarray, shape: input_shape): the value of the tanh activation function for the given input.
    
        """

        return np.tanh(x)

    def derivative(self, x):

        """
        Computes the value of the derivative of the tanh activation function for the given input.

        Parameters
        ----------
        x (np.ndarray, shape: input_shape): the input of the derivative of the tanh activation function.

        Returns (np.ndarray, shape: input_shape): the value of the derivative of the tanh activation function for the given input.
    
        """

        return 1 - np.tanh(x)**2


aliases = {
    "sigm": Sigmoid(),
    "sigmoid": Sigmoid(),
    "relu": ReLU(),
    "ReLU": ReLU(),
    "tanh": Tanh(),
    "identity": Identity()
}
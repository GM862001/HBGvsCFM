import numpy as np


class RegularizationFunction():

    """
    Regularization Function Base Class.

    Methods to override
    -------------------
    __call__(self, params) -> float:
        Computes the value of the regularization function for the given parameters.
    derivative(self, params) -> np.ndarray:
        Computes the value of the derivative of the regularization function for the given parameters.

    """

    def __call__(self, params):

        """
        Computes the value of the regularization function for the given parameters.

        Attributes
        ----------
        params (np.ndarray, shape: params_shape): the input parameters of the regularization function.

        Returns (float): the value of the regularization function for the given parameters.

        """

        raise NotImplementedError
    
    def derivative(self, params):

        """
        Computes the value of the derivative of the regularization function for the given parameters.

        Attributes
        ----------
        params (np.ndarray, shape: params_shape): the input parameters of the derivative of the regularization function.

        Returns (np.ndarray, shape: params_shape): the value of the derivative of the regularization function for the given parameters.

        """

        raise NotImplementedError


class NullRegularization(RegularizationFunction):

    """
    Null Regularization Function Class.
    A regularization function which is null independently of its input.

    Methods
    -------
    __call__(self, params) -> float:
        Computes the value of the null regularization function for the given parameters.
    derivative(self, params) -> np.ndarray:
        Computes the value of the derivative of the null regularization function for the given parameters.

    """

    def __call__(self, params):

        """
        Computes the value of the null regularization function for the given parameters.

        Attributes
        ----------
        params (np.ndarray, shape: params_shape): the input parameters of the null regularization function.

        Returns (float): the value of the null regularization function for the given parameters.

        """

        return 0
    
    def derivative(self, params):

        """
        Computes the value of the derivative of the null regularization function for the given parameters.

        Attributes
        ----------
        params (np.ndarray, shape: params_shape): the input parameters of the derivative of the null regularization function.

        Returns (np.ndarray, shape: params_shape): the value of the derivative of the null regularization function for the given parameters.

        """

        return np.zeros(params.shape)


class Lasso(RegularizationFunction):

    """
    Lasso Regularization Function Class.

    Attributes
    ----------
    l1 (float, default: 0.01): Lasso regularization coefficient.

    Methods
    -------
    __init__(self, l1 = 0.01):
        Initializes the Lasso regularization function coefficient.
    __call__(self, params) -> float:
        Computes the value of the Lasso regularization function for the given parameters.
    derivative(self, params) -> np.ndarray:
        Computes the value of the derivative of the Lasso regularization function for the given parameters.

    """

    def __init__(self, l1 = 0.01):
        
        """
        Initializes the Lasso regularization function coefficient.

        Parameters
        ----------
        l1 (float, default: 0.01): Lasso regularization coefficient.

        """

        self.l1 = l1

    def __call__(self, params):

        """
        Computes the value of the Lasso regularization function for the given parameters.

        Attributes
        ----------
        params (np.ndarray, shape: params_shape): the input parameters of the Lasso regularization function.

        Returns (float): the value of the Lasso regularization function for the given parameters.

        """

        return self.l1 * np.sum(np.abs(params))

    def derivative(self, params):

        """
        Computes the value of the derivative of the Lasso regularization function for the given parameters.

        Attributes
        ----------
        params (np.ndarray, shape: params_shape): the input parameters of the derivative of the Lasso regularization function.

        Returns (np.ndarray, shape: params_shape): the value of the derivative of the Lasso regularization function for the given parameters.

        """

        return self.l1 * np.sign(params)


class Ridge(RegularizationFunction):

    """
    Ridge Regularization Function Class.

    Attributes
    ----------
    l2 (float, default: 0.01): Ridge regularization coefficient.

    Methods
    -------
    __init__(self, l2 = 0.01):
        Initializes the Ridge regularization function coefficient.
    __call__(self, params) -> float:
        Computes the value of the Ridge regularization function for the given parameters.
    derivative(self, params) -> np.ndarray:
        Computes the value of the derivative of the Ridge regularization function for the given parameters.

    """

    def __init__(self, l2 = 0.01):
        
        """
        Initializes the Ridge regularization function coefficient.

        Parameters
        ----------
        l2 (float, default: 0.01): Ridge regularization coefficient.

        """

        self.l2 = l2

    def __call__(self, params):

        """
        Computes the value of the Ridge regularization function for the given parameters.

        Attributes
        ----------
        params (np.ndarray, shape: params_shape): the input parameters of the Ridge regularization function.

        Returns (float): the value of the Ridge regularization function for the given parameters.

        """

        return 0.5 * self.l2 * np.sum(np.square(params))

    def derivative(self, params):

        """
        Computes the value of the derivative of the Ridge regularization function for the given parameters.

        Attributes
        ----------
        params (np.ndarray, shape: params_shape): the input parameters of the derivative of the Ridge regularization function.

        Returns (np.ndarray, shape: params_shape): the value of the derivative of the Ridge regularization function for the given parameters.

        """

        return self.l2 * params


class ElasticNet(RegularizationFunction):

    """
    Elastic Net Regularization Function Class.

    Attributes
    ----------
    lasso (Lasso, default: 0): Lasso term of the elastic net regularization function.
    ridge (Ridge, default: 0): Ridge term of the elastic net regularization function.

    Methods
    -------
    __init__(self, l1 = 0.01, l2 = 0.01):
        Initializes the Lasso and Ridge coefficients of the elastic net regularization function.
    __call__(self, params) -> float:
        Computes the value of the elastic net regularization function for the given parameters.
    derivative(self, params) -> np.ndarray:
        Computes the value of the derivative of the elastic net regularization function for the given parameters.

    """

    def __init__(self, l1 = 0.01, l2 = 0.01):
        
        """
        Initializes the Lasso and Ridge coefficients of the elastic net regularization function.

        Parameters
        ----------
        l1 (float, default: 0.01): Lasso regularization coefficient.
        l2 (float, default: 0.01): Ridge regularization coefficient.

        """

        self.lasso = Lasso(l1)
        self.ridge = Ridge(l2)

    def __call__(self, params):

        """
        Computes the value of the elastic net regularization function for the given parameters.

        Attributes
        ----------
        params (np.ndarray, shape: params_shape): the input parameters of the elastic net regularization function.

        Returns (float): the value of the elastic net regularization function for the given parameters.

        """

        return self.lasso(params) + self.ridge(params)

    def derivative(self, params):

        """
        Computes the value of the derivative of the elastic net regularization function for the given parameters.

        Attributes
        ----------
        params (np.ndarray, shape: params_shape): the input parameters of the derivative of the elastic net regularization function.

        Returns (np.ndarray, shape: params_shape): the value of the derivative of the elastic net regularization function for the given parameters.

        """

        return self.lasso.derivative(params) + self.ridge.derivative(params)


aliases = {
    "null": NullRegularization(),
    "Null": NullRegularization(),
    "zero": NullRegularization(),
    "Zero": NullRegularization(),
    "l1": Lasso(),
    "L1": Lasso(),
    "lasso": Lasso(),
    "Lasso": Lasso(),
    "l2": Ridge(),
    "L2": Ridge(),
    "ridge": Ridge(),
    "Ridge": Ridge(),
    "elastic_net": ElasticNet(),
    "ElasticNet": ElasticNet()
}
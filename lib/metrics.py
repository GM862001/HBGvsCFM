import numpy as np
from sklearn.metrics import accuracy_score


class Metric():

    """
    Metric Base Class.
    
    Methods to override
    -------------------
    __call__(self, y_true, y_pred) -> Float:
        Computes the value of the metric for the given set of predictions and ground truth.
        
    """

    def __call__(self, y_true, y_pred):

        """
        Computes the value of the metric for the given set of predictions and ground truth.

        Parameters
        ----------
        y_true (np.ndarray, shape: (n_samples, n_output_features)): the ground truth values of the target variable.
        y_pred (np.ndarray, shape: (n_samples, n_output_features)): the predicted values of the target variable.

        Returns (float): the value of the metric for the given set of predictions and ground truth.

        """

        raise NotImplementedError


def accuracy(y_true, y_pred):

    if y_true.shape[1] > 1: # One-hot encoded predictions and targets
        true_classes = np.argmax(y_true, axis = 1)
        predicted_classes = np.argmax(y_pred, axis = 1)
    else: # Binary classification without one-hot encoding
        true_classes = y_true
        predicted_classes = np.round(y_pred)
    return accuracy_score(true_classes, predicted_classes)

class Accuracy(Metric):

    """
    Accuracy Class.

    Methods
    -------
    __call__(self, y_true, y_pred) -> Float:
        Computes the accuracy of the given set of predictions and ground truth.
    __repr__(self) -> str:
		Returns a readable representation of the accuracy.
    """


    def __call__(self, y_true, y_pred):

        """
        Computes the accuracy of the given set of predictions and ground truth.

        Parameters
        ----------
        y_true (np.ndarray, shape: (n_samples, n_output_features)): the ground truth values of the target variable.
        y_pred (np.ndarray, shape: (n_samples, n_output_features)): the predicted values of the target variable.

        Returns (float): the accuracy of the given set of predictions and ground truth.

        """

        return accuracy(y_true, y_pred)
    
    def __repr__(self):

        """
		Returns a readable representation of the accuracy.

		Returns (str): a readable representation of the accuracy.

		"""

        return "accuracy"



metrics_aliases = {
    "acc": Accuracy(),
    "accuracy": Accuracy()
}


class ErrorFunction(Metric):

    """
    Error Function Base Class. 

    Methods to override
    -------------------
    Metric.__call__(self, y_true, y_pred) -> Float:
        Computes the value of the error function for the given set of predictions and ground truth.
    derivative(self, y_true, y_pred) -> np.ndarray:
        Computes the value of the derivative of the error function with respect to the predicted values of the target variable for the given set of predictions and ground truth.

    """

    def derivative(self, y_true, y_pred):

        """
        Computes the value of the derivative of the error function with respect to the predicted values of the target variable for the given set of predictions and ground truth.

        Parameters
        ----------
        y_true (np.ndarray, shape: (n_samples, n_output_features)): the ground truth values of the target variable.
        y_pred (np.ndarray, shape: (n_samples, n_output_features)): the predicted values of the target variable.

        Returns (np.ndarray, shape: (n_samples, n_output_features)): the value of the derivative of the error function with respect to the predicted values of the target variable for the given set of predictions and ground truth.

        """

        raise NotImplementedError


def mse(y_true, y_pred):

    return 1 / 2 * np.mean(np.square(y_pred - y_true))

class MSE(ErrorFunction):

    """
    Mean Squared Error (MSE) Class.

    Methods
    -------
    __call__(self, y_true, y_pred) -> Float:
        Computes the MSE for the given set of predictions and ground truth.
    derivative(self, y_true, y_pred) -> np.ndarray:
        Computes the value of the derivative of the MSE with respect to the predicted values of the target variable for the given set of predictions and ground truth.
    __repr__(self) -> str:
		Returns a readable representation of the MSE.

    """

    def __call__(self, y_true, y_pred):

        """
        Computes the MSE for the given set of predictions and ground truth.

        Parameters
        ----------
        y_true (np.ndarray, shape: (n_samples, n_output_features)): the ground truth values of the target variable.
        y_pred (np.ndarray, shape: (n_samples, n_output_features)): the predicted values of the target variable.

        Returns (float): the MSE for the given set of predictions and ground truth.

        """

        return mse(y_true, y_pred)

    def derivative(self, y_true, y_pred):

        """
        Computes the value of the derivative of the MSE with respect to the predicted values of the target variable for the given set of predictions and ground truth.

        Parameters
        ----------
        y_true (np.ndarray, shape: (n_samples, n_output_features)): the ground truth values of the target variable.
        y_pred (np.ndarray, shape: (n_samples, n_output_features)): the predicted values of the target variable.

        Returns (np.ndarray, shape: (n_samples, n_output_features)): the value of the derivative of the MSE with respect to the predicted values of the target variable for the given set of predictions and ground truth.

        """

        return (y_pred - y_true) / y_true.shape[0]

    def __repr__(self):

        """
		Returns a readable representation of the MSE.

		Returns (str): a readable representation of the MSE.

		"""

        return "MSE"



def mee(y_true, y_pred):

    return np.mean(np.linalg.norm(y_pred - y_true, axis=1))

class MEE(ErrorFunction):

    """
    Mean Squared Error (MEE) Class.

    Methods
    -------
    __call__(self, y_true, y_pred) -> Float:
        Computes the MEE for the given set of predictions and ground truth.
    derivative(self, y_true, y_pred) -> np.ndarray:
        Computes the value of the derivative of the MEE with respect to the predicted values of the target variable for the given set of predictions and ground truth.
    __repr__(self) -> str:
		Returns a readable representation of the MEE.

    """

    def __call__(self, y_true, y_pred):

        """
        Computes the MEE for the given set of predictions and ground truth.

        Parameters
        ----------
        y_true (np.ndarray, shape: (n_samples, n_output_features)): the ground truth values of the target variable.
        y_pred (np.ndarray, shape: (n_samples, n_output_features)): the predicted values of the target variable.

        Returns (float): the MEE for the given set of predictions and ground truth.

        """

        return mee(y_true, y_pred)

    def derivative(self, y_true, y_pred):

        """
        Computes the value of the derivative of the MEE with respect to the predicted values of the target variable for the given set of predictions and ground truth.

        Parameters
        ----------
        y_true (np.ndarray, shape: (n_samples, n_output_features)): the ground truth values of the target variable.
        y_pred (np.ndarray, shape: (n_samples, n_output_features)): the predicted values of the target variable.

        Returns (np.ndarray, shape: (n_samples, n_output_features)): the value of the derivative of the MEE with respect to the predicted values of the target variable for the given set of predictions and ground truth.

        """

        return (y_pred - y_true) / np.linalg.norm(y_pred - y_true, axis = 1, keepdims = True) / y_true.shape[0]

    def __repr__(self):

        """
		Returns a readable representation of the MEE.

		Returns (str): a readable representation of the MEE.

		"""

        return "MEE"


error_functions_aliases = {
    "MSE": MSE(),
    "mse": MSE(),
    "MEE": MEE(),
    "mee": MEE()
}

metrics_aliases.update(error_functions_aliases)

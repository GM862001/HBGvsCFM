import numpy as np

from .metrics import Metric, metrics_aliases


class EarlyStopping():

    """
    Early Stopping Class.

    Attributes
    ----------
    metric (str or Metric, default: "loss"): an alias for the metric to monitor, the metric itself, or "loss".
    target_dataset ("train" or "test", default: "test"): whether the monitored metric must be measured on the training or on the test set.
    mode ("min" or "max", default: "min"): whether the monitored metric is to be minimized or maximized.
    tolerance (float, 1e-4): the minimum change in the monitored metric to qualify as an improvement.
    patience (int, default: 10): the number of epochs with no improvement after which training will be stopped.
    _best_metric_value (float): the best value obtained for the monitored metric since the start of the training.
    _best_epoch (int): the training epoch in wich the best value of the monitored metric was obtained.
    _n_epochs (int): the current epoch of training.
    _not_improving_epochs (int): the number of training epochs elapsed since the last improvement of the monitored metric.

    Methods
    -------
    __init__(self, metric = "loss", target_dataset = "test", mode = "min", tolerance = 1e-4, patience = 10, verbose = True):
        Initializes the hyperparameters of the early stopping.
    initialize(self):
        Initializes the parameters of the early stopping before the start of training.
    on_epoch_end(self, loss, y_true, y_pred, params) -> bool:
        Checks the trend of the monitored metric at the end of each training epoch.
    __repr__(self):
        Returns a readable representation of the EarlyStopping.

    """

    def __init__(self, metric = "loss", target_dataset = "test", mode = "min", tolerance = 1e-4, patience = 10):

        """
        Initializes the hyperparameters of the early stopping.

        Parameters
        ----------
        metric (str or Metric, default: "loss"): an alias for the metric to monitor, the metric itself, or "loss".
        target_dataset ("train" or "test", default: "test"): whether the monitored metric must be measured on the training or on the test set.
        mode ("min" or "max", default: "min"): whether the monitored metric is to be minimized or maximized.
        tolerance (float, 1e-4): the minimum change in the monitored metric to qualify as an improvement.
        patience (int, default: 10): the number of epochs with no improvement after which training will be stopped.

        """

        if target_dataset not in ["train", "test"]:
            raise ValueError("The target dataset must be either 'train' or 'test'.")
        self.target_dataset = target_dataset

        if type(metric) == str:
            if metric in metrics_aliases:
                self.metric = metrics_aliases[metric]()
            elif metric == "loss":
                self.metric = metric
            else:
                raise ValueError("Unknown metric.")
        elif issubclass(type(metric), Metric):
            self.metric = metric
        else:
            raise ValueError("The metric must be a Metric object or as string alias for it, or 'loss'.")
        
        if mode not in ["max", "min"]:
            raise ValueError("The mode must be either 'max' or 'min'.")
        self.mode = mode

        self.tolerance = tolerance
        self.patience = patience

    def initialize(self):

        """
        Initializes the parameters of the early stopping before the start of training.

        """

        self._best_metric_value = np.infty if self.mode == "min" else -np.infty
        self._n_epochs = 0
        self._not_improving_epochs = 0

    def on_epoch_end(self, loss, y_true, y_pred, params):

        """
        Checks the trend of the monitored metric at the end of each training epoch.

        Parameters
        ----------
        loss (float): the loss at the current epoch of training for the target dataset.
        y_true (np.ndarray, shape: (n_samples, n_output_features)): the ground truth values of the target variable for the target dataset.
        y_pred (np.ndarray, shape: (n_samples, n_output_features)): the predicted values of the target variable for the target dataset.
        params (list): the current parameters of the model.

        Returns (bool): True if training has to stop, False otherwise.

        """

        self._n_epochs  += 1

        if self.metric == "loss":
            metric_value = loss
        else:
            metric_value = self.metric(y_true, y_pred)

        if (self.mode == "min" and metric_value < self._best_metric_value - self.tolerance) or (self.mode == "max" and metric_value > self._best_metric_value + self.tolerance):
            self._best_metric_value = metric_value
            self._not_improving_epochs = 0
            self._best_epoch = self._n_epochs
            self._best_params = params
        else:
            self._not_improving_epochs += 1
            if self._not_improving_epochs == self.patience:
                return True
        return False

    def __repr__(self):

        """
        Returns a readable representation of the EarlyStopping.

        """

        return f"EarlyStopping(tolerance: {self.tolerance}, patience: {self.patience})"

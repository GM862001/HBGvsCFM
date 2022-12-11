import numpy as np


class Optimizer():

	"""
	Optimizer Base Class.

    Base Methods
	------------
	initialize(self):
		Initializes the parameters of the optimizer before the start of training, if needed.
	after_forward_propagation(self, loss):
		Updates the parameters of the optimizer at the end of each forward pass, if needed.
	on_epoch_end(self):
		Updates the parameters of the optimizer at the end of each epoch of training, if needed.

	Methods to override
	-------------------
	optimize(self, grad_params) -> np.ndarray:
		Performs an optimization step by computing an update of the parameters to optimize.

	"""

	def initialize(self):

		"""
		Initializes the parameters of the optimizer before the start of training, if needed.
		
		"""

		pass

	def optimize(self, grad_params):

		"""
		Performs an optimization step by computing an update of the parameters to optimize.

		Parameters
		----------
		grad_params (np.ndarray, shape: params_shape): the value of the derivative of the loss function with respect to the parameters to optimize.

		Returns (np.ndarray, shape: params_shape): the parameters update.

		"""

		raise NotImplementedError

	def after_forward_propagation(self, loss):

		"""
		Updates the parameters of the optimizer at the end of each forward pass, if needed.

		Parameters
		----------
		loss (float): the value of the loss at the current epoch of training.

		"""

		pass

	def on_epoch_end(self):

		"""
		Updates the parameters of the optimizer at the end of each training epoch, if needed.

		"""

		pass


class HBG(Optimizer):

	"""
	Heavy Ball Gradient (HBG) Optimizer Class.

	Attributes
	----------
	alpha (float, default: 0.1): the coefficient for the stepsize in the direction of the gradient of the loss.
	beta (float, default: 0.5): the coefficient for the stepsize in the direction of the last update of the parameters (momentum coefficient).
	_last_params_update (np.ndarray, shape: params_shape): the last update of the parameters to optimize.

	Methods
	-------
	__init__(self, alpha = 0.1, beta = 0.5):
		Initializes the hyperparameters of the HBG optimizer.
	initialize(self):
		Initializes the parameters of the HBG optimizer before the start of training.
	optimize(self, grad_params) -> np.ndarray:
		Performs a step of HBG by computing an update of the parameters to optimize.
	__repr__(self) -> str:
		Returns a readable representation of the HBG optimizer.

	"""

	def __init__(self, alpha = 0.1, beta = 0.5):

		"""
		Initializes the hyperparameters of the HBG optimizer.

		Parameters
		----------
		alpha (float, default: 0.1): the coefficient for the stepsize in the direction of the gradient of the loss.
		beta (float, default: 0.5): the coefficient for the stepsize in the direction of the last update of the parameters (momentum coefficient).

		"""

		self.alpha = alpha
		self.beta = beta
	
	def initialize(self):

		"""
		Initializes the parameters of the HBG optimizer before the start of training.
		
		"""

		self._last_params_update = 0

	def optimize(self, grad_params):

		"""
		Performs a step of HBG by computing an update of the parameters to optimize.

		Parameters
		----------
		grad_params (np.ndarray, shape: params_shape): the value of the derivative of the loss function with respect to the parameters to optimize.

		Returns (np.ndarray, shape: params_shape): the parameters update.

		"""

		params_update =  - self.alpha * grad_params + self.beta * self._last_params_update
		self._last_params_update = params_update
		return params_update
	
	def __repr__(self):

		"""
		Returns a readable representation of the HBG optimizer.

		Returns (str): a readable representation of the HBG optimizer.

		"""

		return f"HBG Optimizer(alpha = {self.alpha}, beta = {self.beta})"


class CFM(Optimizer):

	"""
	CFM (Camerini-Fratta-Maffioli) Subgradient Optimizer Class.

	Attributes
	----------
	delta_k (str, default: "1/k"): the target level sequence for optimal value approximation. Must be an evaluable expression in k.
	gamma (float, default: 1.5): the deflection hyperparameter.
	best_loss (float, default: None): the optimal value of the loss function or a good estimate of it. If None, a moving target estimate of it is used.
	_n_epochs (int): the current epoch of training.
	_last_update_direction (np.ndarray, shape: params_shape): the direction of the last update of the parameters.
	_current_loss (float): the loss value at the current epoch of training.
	_best_loss_k (float): the least loss value obtained since the start of training up to k-th epoch.

	Methods
	-------
	__init__(self):
		Initializes the hyperparameters of the CFM optimizer.
	initialize(self):
		Initializes the parameters of the HBG optimizer before the start of training.
	optimize(self, grad_params) -> np.ndarray:
		Performs a step of CFM by computing an update of the parameters to optimize.
	after_forward_propagation(self, loss):
		Updates the current and the best values of the loss at the end of each forward pass.
	on_epoch_end(self):
		Updates the number of elapsed epochs.
	__repr__(self) -> str:
		Returns a readable representation of the optimizer.

	"""

	def __init__(self, delta_k = "1/k", gamma = 1.5, best_loss = None):

		"""
		Initializes the hyperparameters of the CFM optimizer.

		Parameters
		----------
		delta_k (str, default: "1/k"): the target level hyperparameter for optimal value approximation. Must be an evaluable expression in k. Used only when best_loss is not provided.
		gamma (float, default: 1.5): the deflection hyperparameter.
		best_loss (float, default: None): the optimal value of the loss function or a good estimate of it. If None, a moving target estimate of it is used.

		"""

		self.gamma = gamma
		self.best_loss = best_loss
		if not best_loss:
			self.delta_k = delta_k

	def initialize(self):
		
		"""
		Initializes the parameters of the CFM optimizer before the start of training.
		
		"""

		self._n_epochs = 1

		if not self.best_loss:
			self._best_loss_k = np.infty

	def optimize(self, grad_params):

		"""
		Performs a step of CFM deflected subgradient by computing an update of the parameters to optimize.

		Parameters
		----------
		grad_params (np.ndarray, shape: params_shape): the value of the derivative of the loss function with respect to the parameters to optimize.

		Returns (np.ndarray, shape: params_shape): the parameters update.

		"""

		if self._n_epochs == 1:
			update_direction = grad_params
		else:
			beta = max(0, - self.gamma * np.sum(self._last_update_direction * grad_params) / np.linalg.norm(self._last_update_direction) ** 2)
			update_direction = grad_params + beta * self._last_update_direction
		self._last_update_direction = update_direction

		if self.best_loss:
			alpha = (self._current_loss - self.best_loss) / np.linalg.norm(update_direction) ** 2 # Polyak stepsize
		else:
			target_level = eval(self.delta_k, {"k": self._n_epochs})
			alpha = (self._current_loss - self._best_loss_k + target_level) / np.linalg.norm(update_direction) ** 2 # Polyak stepsize
		return - alpha * update_direction

	def after_forward_propagation(self, loss):

		"""
		Updates the current and the best values of the loss at the end of each forward pass.

		Parameters
		----------
		loss (float): the value of the loss at the current epoch of training.

		"""

		self._current_loss = loss

		if not self.best_loss:
			if self._current_loss < self._best_loss_k:
				self._best_loss_k = self._current_loss

	def on_epoch_end(self):

		"""
		Updates the number of elapsed epochs.

		"""

		self._n_epochs += 1

	def __repr__(self):

		"""
		Returns a readable representation of the CFM optimizer.

		Returns (str): a readable representation of the CFM optimizer.

		"""

		if self.best_loss:
			return f"CFM Optimizer(best_loss = {self.best_loss}, gamma = {self.gamma})"
		else:
			return f"CFM Optimizer(delta_k = {self.delta_k}, gamma = {self.gamma})"


aliases = {
	"hbg": HBG,
	"HBG": HBG,
	"cfm": CFM,
	"CFM": CFM
}

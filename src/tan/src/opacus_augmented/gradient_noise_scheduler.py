from typing import Dict

from opacus.optimizers import DPOptimizer


class GradientRatioScheduler:

    def __init__(self, optimizer: DPOptimizer, *, last_epoch=-1):
        """
        Args:
            optimizer (DPOptimizer): The DPOptimizer
            *: Any other positional args (this is an abstract base class)
            last_epoch(int): The index of last epoch. Default: -1.
        """
        if not hasattr(optimizer, "noise_multiplier"):
            raise ValueError(
                "NoiseSchedulers require your optimizer to have a .noise_multiplier attr. "
                "Are you sure you are using a DPOptimizer? Those have it added for you."
            )
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.first_epoch_grad = -1
        self.initial_noise_multiplier = optimizer.noise_multiplier

        self.step()

    def state_dict(self) -> Dict:
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.

        """
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict: Dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_noise_multiplier(self, gradient_norm):
        return (gradient_norm/self.first_epoch_grad) * self.initial_noise_multiplier
    
    def set_first_epoch_grad(self, gradient_norm):
        if self.first_epoch_grad != -1:
            raise ValueError("The gradient norm of the first epoch has already been set!")
        else:
            self.first_epoch_grad = gradient_norm

    def step(self, gradient_norm):
        self.last_epoch += 1
        noise_multiplier = self.get_noise_multiplier(gradient_norm)
        self.optimizer.noise_multiplier = noise_multiplier


class GradientRuleScheduler:

    def __init__(self, optimizer: DPOptimizer, *, last_epoch=-1):
        """
        Args:
            optimizer (DPOptimizer): The DPOptimizer
            *: Any other positional args (this is an abstract base class)
            last_epoch(int): The index of last epoch. Default: -1.
        """
        if not hasattr(optimizer, "noise_multiplier"):
            raise ValueError(
                "NoiseSchedulers require your optimizer to have a .noise_multiplier attr. "
                "Are you sure you are using a DPOptimizer? Those have it added for you."
            )
        self.optimizer = optimizer
        self.last_epoch = last_epoch

        self.step()

    def state_dict(self) -> Dict:
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.

        """
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict: Dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_noise_multiplier(self, gradient_norm):
        if gradient_norm >= 12:
            return 5
        elif gradient_norm <= 9:
            return 2
        return 1

    def step(self, gradient_norm):
        self.last_epoch += 1
        noise_multiplier = self.get_noise_multiplier(gradient_norm)
        self.optimizer.noise_multiplier = noise_multiplier
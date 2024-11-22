from typing import Dict

from opacus.optimizers import DPOptimizer

import numpy as np

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
        self.initial_noise_multiplier = optimizer.noise_multiplier
        self.relative_gradients = []

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
        if len(self.relative_gradients) == 0:
            raise ValueError("No relative gradients have been added!")
        return (gradient_norm/np.mean(self.relative_gradients)) * self.initial_noise_multiplier

    def add_relative_gradient(self, gradient_norm):
        self.relative_gradients.append(gradient_norm)

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
        if gradient_norm >= 12.25:
            return 0.375 # equals to noise power 6
        elif gradient_norm >= 11.75:
            return 0.3125 # equals to noise power 5
        return 0.125 # equals to noise power 2

    def step(self, gradient_norm):
        self.last_epoch += 1
        noise_multiplier = self.get_noise_multiplier(gradient_norm)
        self.optimizer.noise_multiplier = noise_multiplier
from abc import ABC, abstractmethod
from torch import Tensor

class BaseAttack(ABC):
    """
    Base class for attacks based on Projected Gradient Descent.

    Since adversarial attacked are defined in a specific "ball", you need to project back
    after applying the attack.
    """
    def __init__(self, original_input: Tensor, attack_rate: float, eps: float) -> None:
        # TODO: docstring
        self._original_input = original_input
        self._attack_rate = attack_rate
        self._eps = eps

    @abstractmethod
    def project_back(self, input: Tensor) -> Tensor:
        """Method that projects the perturbed example to the lp-ball."""
        pass

    @abstractmethod
    def apply_attack_iteration(self, input: Tensor, grad: Tensor) -> Tensor:
        """Method that applies one iteration of attack given the input gradient."""
        # TODO: docstring
        pass

class L2Attack(BaseAttack):
    # TODO: class for l2
    pass

class LinfAttack(BaseAttack):
    # TODO: class for l-\infty
    pass

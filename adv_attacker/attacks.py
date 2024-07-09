from abc import ABC, abstractmethod

import torch
from torch import Tensor


class BaseAttack(ABC):
    """
    Base class for attacks based on Projected Gradient Descent.

    Since adversarial attacked are defined in a specific "ball", you need to project back
    after applying the attack.
    """

    def __init__(self, original_input: Tensor, attack_rate: float, eps: float) -> None:
        """
        Constructor for the `BaseAttack` class.

        :param original_input: The original image.
        :param attack_rate: The attack rate used for the attack:
            x_{new} = x_{orig} + attack_rate \times normalized_gradient.
        :param eps: The maximum norm of attack.
        """
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
        pass


class L2Attack(BaseAttack):
    """Class for l2-attacks."""

    def __init__(self, original_input: Tensor, attack_rate: float, eps: float) -> None:
        super().__init__(original_input, attack_rate, eps)

    def project_back(self, input: Tensor) -> Tensor:
        """
        Project back to the l2-norm by constraining the different to a maximum
        norm of `eps`.
        """
        # compute the difference between x and the original input
        diff = input - self._original_input

        # scale the difference to have a maximum norm of eps
        scaled_diff = diff.renorm(p=2, dim=0, maxnorm=self._eps)

        # add the scaled difference back to the original input
        adjusted_input = self._original_input + scaled_diff

        # clamp the result to the range [0, 1]
        clamped_result = torch.clamp(adjusted_input, 0, 1)

        return clamped_result

    def apply_attack_iteration(self, input: Tensor, grad: Tensor) -> Tensor:
        """Apply one iteration of the attack: x_{new} = x_{orig} + attack_rate \times normalized_gradient."""
        # we need to normalize the gradient first
        batch_size = grad.shape[0]
        flattened_dim = grad.reshape(batch_size, -1)

        # compute the norm along the second dimension
        g_norm_flattened = torch.norm(flattened_dim, dim=1)

        # determine the required number of singleton dimensions to match the original shape
        ld = len(input.shape) - 1
        # reshape the norm result to insert singleton dimensions
        grad_norm = g_norm_flattened.view(-1, *([1] * ld))
        # scale the grad
        normalized_grad = grad / (grad_norm + 1e-16)
        # add the perturbation to the input
        perturbed_input = input + normalized_grad * self._attack_rate
        return perturbed_input


class LinfAttack(BaseAttack):
    """Class for linf-attacks."""

    # TODO: class for l-\infty
    pass

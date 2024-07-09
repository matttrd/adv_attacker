from typing import Optional

import torch
from torch import Tensor
from torch.nn import Module

from adv_attacker.attacks import L2Attack, LinfAttack

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelAttacker(Module):
    # TODO : docstring
    def __init__(
        self,
        model,
        attack_norm: str,
        loss_function: Optional[callable] = None,
        mean: Optional[list | Tensor] = None,
        std: Optional[list | Tensor] = None,
    ):

        super(ModelAttacker, self).__init__()
        self._model = model
        if attack_norm == "l2":
            self._attack_class = L2Attack
        elif attack_norm == "linf":
            self._attack_class = LinfAttack
        else:
            raise ValueError(f"The attacker {attack_norm} is not implemented.")

        # TODO: we may want custom losses
        if loss_function is None:
            # use standard one
            self._loss_function = torch.nn.CrossEntropyLoss(reduction="none").to(
                device=device
            )

        self._mean = mean if isinstance(mean, Tensor) else torch.tensor(mean)
        self._std = std if isinstance(std, Tensor) else torch.tensor(std)
        self._mean = self._mean.to(device=device)
        self._std = self._std.to(device=device)

    def _get_loss_and_output(
        self, input: Tensor, target: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Compute the loss for the batch.

        :param input: The input image tensor.
        :param target: the target class tensor.
        :return: A tuple of output and loss for the input image.
        """
        if self._mean is None:
            mean = torch.tensor([0.0, 0.0, 0.0])
            std = torch.tensor([1.0, 1.0, 1.0])

        mean = self._mean[:, None, None]
        std = self._std[:, None, None]

        norm_input = (input - mean) / std
        output = self._model(norm_input)
        return self._loss_function(output, target), output

    def forward(
        self,
        input: Tensor,
        target_class: Tensor,  # TODO: make this optional to allow to apply non-targeted attacks
        eps: Optional[float] = None,
        attack_rate: float = 0.01,
        num_steps: int = 8,
        skip_adv_attack: bool = False,
    ) -> tuple[Tensor, Tensor]:

        if not skip_adv_attack:
            # TODO: implement forward method for attacker
            assert eps is not None, "`eps` must be a valid float number"
            input = input.to(device=device)
            assert (
                target_class is not None
            ), "Must specify a target class in the current implementation."
            target_class = target_class.to(device=device)

            attacker = self._attack_class(
                original_input=input, eps=eps, attack_rate=attack_rate
            )
            adv_image = input.detach()

            # TODO: run the attack. Steps:
            # 1. put the model in eval (avoid to update batchnorm stats)
            # 2. Attack
            # 3. Return the image

            self._model.eval()

            # TODO: add change of random noise before starting to avoid bad gradients
            # TODO: add multiple trials if get the best example
            # TODO: choose the best instead of the last one

            # iterate for num_steps
            for _ in range(num_steps):
                # we need to copy the image and detach to avoid the gradient propagation
                adv_image = adv_image.clone().detach()
                # we need to compute the grad so we must activate it
                adv_image = adv_image.requires_grad_(True)
                # output = self._model(adv_image)
                # batch_loss = self._loss_function(output, target_class)
                batch_loss, output = self._get_loss_and_output(
                    input=adv_image, target=target_class
                )
                loss = batch_loss.mean()
                # TODO: modify here if want to do non-target attacks
                (grad,) = torch.autograd.grad(-loss, [adv_image])

                # now we need to apply the attack but make sure we don't compute any gradient (we may not need this)
                with torch.no_grad():
                    adv_image = attacker.apply_attack_iteration(
                        input=adv_image, grad=grad
                    )
                    # project back to make sure to stay in the correct interval
                    adv_image = attacker.project_back(adv_image)

        else:
            adv_image = input
            output = self._model(adv_image)

        return output, adv_image

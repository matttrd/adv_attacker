import unittest
import numpy
from PIL import Image
from adv_attacker import ModelAttacker
import torch

from torchvision.models import resnet18, ResNet18_Weights # import a small model
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
preprocess = weights.transforms()

# determine the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

class TestAdversarialAttack(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # TODO: initialization
        #
        cls.rng = numpy.random.default_rng()
        cls.l2_attacker = ModelAttacker(model=model, attack_norm='l2')
        # load a test image
        image_path = 'tests/images/dog_image.jpeg'  # Replace with your image path
        img = Image.open(image_path)
        img_torch = preprocess(img)
        # denormalize image
        # TODO: add normalization inside the core functions to avoid this step
        cls.img_torch = img_torch * torch.tensor(preprocess.std)[:, None, None] + torch.tensor(preprocess.mean)[:, None, None]
        cls.img_torch =  cls.img_torch.to(device=device)


    def test_api(self):
        # unsqueeze image
        # TODO: improve the code to avoid this step
        img = self.img_torch.unsqueeze(0)
        # test it returns the same image
        target_class = torch.tensor([0])
        output, adv_image = self.l2_attacker(input=img, skip_adv_attack=True, target_class=target_class)
        # must be the same 
        torch.testing.assert_close(img, adv_image)
        self.assertEqual(output.shape, (1, 1000))

    def test_single_image_l2_attack(self):
        img = self.img_torch.unsqueeze(0)
        target_class = torch.tensor([0])
        
        # check with eps = 0.
        output, adv_image = self.l2_attacker(input=img, eps=0.00005, target_class=target_class)
        # check the relative difference is very small
        self.assertLess(((adv_image - img).norm() / img.norm()).item(), 1e-4)
        # check it is a Labrador retriever (see here: https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/)
        self.assertEqual(output.argmax().item(), 208)

        # check with higher eps
        output, adv_image = self.l2_attacker(input=img, eps=4, target_class=target_class, attack_rate=0.5)
        # check we reached the correct target class
        self.assertEqual(output.argmax().item(), 0)

        # choose another target class: 38 - banded gecko
        target_class = torch.tensor([38])
        output, adv_image = self.l2_attacker(input=img, eps=4, target_class=target_class, attack_rate=0.5)
        self.assertEqual(output.argmax().item(), 38)
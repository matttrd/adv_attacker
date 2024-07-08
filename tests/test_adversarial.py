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
        cls.img_torch =  cls.img_torch.to_device(device=device)

    def test_api(self):
        # unsqueeze image
        # TODO: improve the code to avoid this step
        img = self.img_torch.unsqueeze(0)
        # test it returns the same image
        output, input = self.l2_attacker(input=img, skip_adv_attack=True)
        # must be the same 
        torch.testing.assert_close(input, img)


    def test_single_image(self):
        
        img = self.img_torch.unsqueeze(0)
        output, input = self.l2_attacker()

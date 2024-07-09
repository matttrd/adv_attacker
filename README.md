# adv_attacker
Library for applying adversarial attack

## How to install it

1. **Clone the Repository**

    ```sh
    git clone https://github.com/matttrd/adv_attacker.git
    cd adv_attacker
    ```

2. **Install the Package with Poetry**

    Run the following command to install the dependencies and the package itself:

    ```sh
    poetry install
    ```

## Quickstart
The following example shows how to generate an adversarial attack for a test image.

```python

from PIL import Image 
from torchvision.models import resnet18, ResNet18_Weights # import a small model

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
preprocess = weights.transforms()

# determine the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# test with a sample image
image_path = 'tests/images/dog_image.jpeg'  # Replace with your image path
img = Image.open(image_path)
# pre-process the image
img_torch = preprocess(img)

# since the attacker must work in the range [0,1] but the image is
# mormalized with ImageNet statistics you need to de-normalized first
mean = torch.tensor(preprocess.mean)[:, None, None]
std = torch.tensor(preprocess.std)[:, None, None]
denorm_img = (img_torch * std + mean).to(device=device)
# we need to add the batch dimension on the input
denorm_img = denorm_img.unsqueeze(0)

# instantiate the l2 attacker (provide the stats to make sure the model receives the normalized image)
l2_attacker = ModelAttacker(model=model, attack_norm='l2', mean=preprocess.mean, std=preprocess.std)

# define the target class:  38 - banded gecko
target_class = torch.tensor([38])
output, adv_image = l2_attacker(input=denorm_img, eps=1, target_class=target_class, attack_rate=0.5)
print(f"Predicted class is {output.argmax().item()}")
```

## Configuration
# TODO: description

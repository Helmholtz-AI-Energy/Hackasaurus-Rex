import torch
import torchvision

_default_means = [130.0, 135.0, 135.0, 118.0, 118.0]
_default_vars = [44.0, 40.0, 40.0, 30.0, 21.0]

transformations = torch.nn.Sequential(
    torchvision.transforms.v2.ToTensor(),
    torchvision.transforms.Normalize(_default_means, _default_vars),
    torchvision.transforms.v2.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.v2.RandomVerticalFlip(p=0.5),
    torchvision.transforms.v2.RandomRotation(90),
)

transformations = torch.jit.script(transformations)

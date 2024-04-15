import numpy as np
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.utils import save_image,make_grid
import yaml
import dlib
import cv2
from PyQt5.QtWidgets import QMessageBox
from torchvision.transforms import Resize


def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    if x.min() < 0:
        out = (x + 1) / 2
        return out.clamp_(0, 1)
    else:
        return x



def get_config(config):
    with open(config, 'r') as stream:
        return yaml.safe_load(stream)


def Image2tensor(imagepath,process=False,resize=256,device='cuda'):
    img = Image.open(imagepath)
    transform = []
    transform.append(T.ToTensor())
    if len(img.split()) == 3:
        transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    else:
       transform.append(T.Normalize(mean=0.5, std=0.5))
    if process:
        transform.append(T.Resize([resize,resize]))
    transform = T.Compose(transform)
    img = torch.unsqueeze(transform(img),dim=0).to(device)
    return img


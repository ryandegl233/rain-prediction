import torch
from kornia.morphology import dilation, erosion, opening, closing
from kornia.filters import GaussianBlur2d

def test_opening():
    img = torch.rand(1, 1, 256, 256)
    # mask = img < 0.4
    kernel = torch.ones(3, 3)
    img_open = opening(img, kernel=kernel)
    pass


test_opening()

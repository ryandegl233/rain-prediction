import numpy as np
import torch as th
from kornia.filters import GaussianBlur2d
from skimage.filters import gaussian


def gaussian_data(
    data: np.ndarray | th.Tensor,  # rain data [h, w]
    sigma: float = 1.5,
    unchanged_amp=True,
    n_times: int = 1,
    use_cuda=False,
):
    orig_tensor = th.is_tensor(data)

    if use_cuda or orig_tensor:
        data = th.as_tensor(data).float()
        if use_cuda:
            data = data.cuda()
        data = data.view(1, 1, data.shape[-2], data.shape[-1])
        gss = GaussianBlur2d(5, (sigma, sigma))

    for i in range(n_times):
        max_d = data.max()
        if not use_cuda and not orig_tensor:
            data = gaussian(data, sigma=sigma)
        else:
            data = gss(data)

        if unchanged_amp:
            data = data * max_d / data.max()

    if use_cuda or orig_tensor:
        data = data[0, 0]
        if not orig_tensor:
            data = data.cpu().numpy()

    return data

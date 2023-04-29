import torch
import torch.nn as nn
import numpy as np

import os
import imageio
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from new_datasets import data_preprocessing
from model import Voxels,Nerf
from ml_helpers import training
import rendering

data_set_path = './Dataset'
mode = 'test'
target_size = (400,400)
dataset = data_preprocessing(data_set_path,mode,target_size=target_size)


test_o, test_d, target_px_values,total_data = dataset.get_rays()

device='cuda'
tn=2
tf=6

model =torch.load('model_nerf').to(device)


def mse2psnr(mse):
    return 20 * np.log10(1 / np.sqrt(mse))


@torch.no_grad()
def test(model, o, d, tn, tf, nb_bins=100, chunk_size=10, H=400, W=400, target=None):
    o = o.chunk(chunk_size)
    d = d.chunk(chunk_size)

    image = []
    for o_batch, d_batch in zip(o, d):
        img_batch = rendering.rendering(model, o_batch, d_batch, tn, tf, nb_bins=nb_bins, device=o_batch.device)
        image.append(img_batch)  # N, 3
    image = torch.cat(image)
    image = image.reshape(H, W, 3).cpu().numpy()

    if target is not None:
        mse = ((image - target) ** 2).mean()
        psnr = mse2psnr(mse)

    if target is not None:
        return image, mse, psnr
    else:
        return image

img, mse, psnr = test(model, torch.from_numpy(test_o[0]).to(device).float(), torch.from_numpy(test_d[0]).to(device).float(),
                tn, tf, nb_bins=100, chunk_size=10, target=target_px_values[0].reshape(400, 400, 3))
plt.imshow(img)
print(psnr)
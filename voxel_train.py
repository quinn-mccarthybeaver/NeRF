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

torch.cuda.empty_cache()

data_set_path = '/home/eiyike/Desktop/phdproject/MY_NERf2222/new_code_update1/Dataset'


mode = 'train'

dataset = data_preprocessing(data_set_path,mode,target_size=(400,400))

o, d, target_px_values,total_data = dataset.get_rays()

batch_size = 1024

dataloader = DataLoader(
    torch.cat(
              (torch.from_numpy(o).reshape(-1, 3),
               torch.from_numpy(d).reshape(-1, 3),
               torch.from_numpy(target_px_values).reshape(-1, 3)
               ), dim=1),
              batch_size=batch_size, shuffle=True)       # (-1,3)  means N*H*W , 3


device = 'cuda'

tn = 2
tf = 6
nb_epochs = 1
lr = 1e-3
gamma = .5
nb_bins = 100

model = Voxels(scale=3, device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma=gamma)

nb_epochs = 1
training_loss = training(model, optimizer, scheduler, tn, tf, nb_bins, nb_epochs, dataloader, device=device)
plt.plot(training_loss)
plt.show()


img = rendering.rendering(model, torch.from_numpy(o[0]).to(device), torch.from_numpy(d[0]).to(device),
                tn, tf, nb_bins=100, device=device)
plt.imshow(img.reshape(400, 400, 3).data.cpu().numpy())
from PIL import Image
from einops import rearrange
from VGG import VGG_decoder
import torchvision.transforms as standard_transforms
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
from torchvision import models
import numpy as np
from lwcc import LWCC
from scipy.io import loadmat
import os

if __name__=='__main__':
    file="./DATA/shanghairaw/ShanghaiTech/part_A/train_data/images/IMG_2.jpg"
    
    density=np.random.random(size=(125,566,3))
    np.save(file="data.npy", arr=density)
    
    image_np = np.load(file="data.npy")
    
    print(np.sum(image_np-density))
    
    
    
    
    # count, density = LWCC.get_count(file, return_density=True)
    #
    # plt.imshow(density)
    # plt.show()
    
    
    
    file_gt = file.replace('images', 'ground-truth').replace('IMG', 'GT_IMG').replace('jpg', 'mat')
    image = Image.open(file).convert('RGB')
    net=VGG_decoder()
    net.eval()
    loaded_state_dict=torch.load("./weights/VGGPre.pth")
    new_state_dict = {k.replace('CCN.', ''): v for k, v in loaded_state_dict.items()}
    net.load_state_dict(new_state_dict)
    mean_std = ([0.376973062754, 0.376973062754, 0.376973062754],[0.206167116761, 0.206167116761, 0.206167116761])
    img_transform = standard_transforms.Compose([
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(*mean_std)
        ])

    with torch.no_grad():
        img = img_transform(image)
        print("img.shape",img.shape)
        img = img.cuda()
        net = net.cuda()
        pred_map = net(img)
        print("pred_map.shape", pred_map.shape)

        pred_map = pred_map.repeat(3, 1, 1)
        npimg = pred_map.cpu().numpy()
        plt.imshow(image)
        plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap="jet",  alpha=0.7)
        
        # plt.imshow(image)
        plt.show()
        





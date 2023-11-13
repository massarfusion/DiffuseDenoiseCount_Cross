import numpy
from scipy.io import loadmat
import torch
from  torch import nn
import  math

#
# file='DATA/shanghairaw/ShanghaiTech/part_B/train_data/images/IMG_5.jpg'
# file = file.replace('images','ground-truth').replace('IMG','GT_IMG').replace('jpg','mat')
# loadedMat=loadmat(file)
# locations = loadmat(file)['image_info'][0][0]['location'][0][0]
# l2 = numpy.array(locations)
# for x,y in locations:
#     x, y = int(x), int(y)
#     print("x is  {} and y is {}".format(x,y))

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
    
s1 = SinusoidalPositionEmbeddings(1000)
input= torch.randn(size=(25,))
output = s1(input)
print(output.shape, "output.shape")

inside = torch.linspace(10,114,25)
print(inside.shape)
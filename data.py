import torch
import scipy
from scipy.io import loadmat


def get_data(path):
    img_data = scipy.io.loadmat(path)["data"]  
    img_data = img_data.reshape((img_data.shape[1],img_data.shape[0]))
    print(f"Shape of a data point: {img_data.shape}")
    print(f"Example data {img_data[0:1].shape}")
    return torch.Tensor(img_data)
def get_minibatch(x,batch_size ,device='cpu'):
    indices = torch.randperm(x.shape[0])[:batch_size]
    return x[indices].reshape(batch_size, -1).to(device)

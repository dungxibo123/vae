import torch
import scipy
from scipy.io import loadmat



def get_minibatch(x,batch_size,device='cpu'):
    indices = torch.randperm(x.shape[0])[:batch_size]
    return x[indices].reshape(batch_size, -1).to(device)
def get_data(path, noise=None,nums=25000):
    img_data = scipy.io.loadmat(path)["data"].T / 255.
    img_data = torch.Tensor(img_data)
    img_data = get_minibatch(img_data,nums)
    img_data = img_data.to(device)
    distri = torch.zeros(img_data.shape)
    
    print(f"Shape of a data point: {img_data.shape}")
    print(f"Example data {img_data[0:1].shape}")
    return img_data

def add_noise(data, std=1/10.,device='cpu'):
    distri = torch.normal(torch.zeros(data.shape),torch.ones(data.shape) * std).to(device)
    return data + distri
    #print(distri)

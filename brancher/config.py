"""
Config
---------
Set device to 'cpu' or 'cuda:index'
Default: cuda:0 if available, otherwise cpu

"""
import torch

default_device = 'cpu'

def set_device(device_):
    global device
    device_ = device_.lower()
    if isinstance(device_, str):
        if 'cuda' in device_:
            assert torch.cuda.is_available(), "Cuda requested but not available"
            device = torch.device(device_)
        elif device_ == 'cpu':
            device = torch.device(device_)
        elif device_ == 'gpu':
            assert torch.cuda.is_available(), "Cuda requested but not available"
            device = torch.device('cuda:0')
        else:
            raise ValueError("Device is not recongnized")
    elif isinstance(device_, int):
        device = torch.device(device_)

if 'device' not in locals():
    set_device(default_device)


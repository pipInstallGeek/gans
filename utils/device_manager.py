# utils/device_manager.py
# DeviceManager: Handles device selection and GPU memory management

import torch

class DeviceManager:
    def __init__(self, device=None):
        self.device = device or (torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    def is_cuda(self):
        return self.device.type == 'cuda'

    def empty_cache(self):
        if self.is_cuda():
            torch.cuda.empty_cache()

    def to_device(self, obj):
        return obj.to(self.device)

    def __str__(self):
        return str(self.device)

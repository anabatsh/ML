import torch
import torchvision.transforms as transforms

class ToTensor():
    def __init__(self):
        self.tr = transforms.ToTensor()

    def __call__(self, img):
        return self.tr(img)

class DeNoise():

    def __call__(self, img):
        img = img.to(device)
        fft = torch.fft.rfft2(img)
        fft = torch.fft.irfft2(fft)
        return fft

class ToDevice():
    def __init__(self, device):
        self.device = device

    def __call__(self, img):
        img = img.to(self.device)
        return img

class Normalize():
    def __call__(self, img):
        max = torch.max(img)
        min = torch.min(img)
        mean = (max - min) / 2
        std = max - mean

        img = (img - mean) / std
        img = (img * 127.5 + 127.5)
        return img

class Compose():
    def __init__(self, *args):
        self.compose = transforms.Compose(*args)

    def __call__(self, img):
        return self.compose(img)

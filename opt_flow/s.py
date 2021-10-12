import torch
import cv2
import joblib
from joblib import Parallel, delayed
import glob
import natsort

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DeNoise():

    def __call__(self, img):
        img = img.to(device)
        fft = torch.fft.rfft2(img)
        fft = torch.fft.irfft2(fft)
        return fft

class ToDevice():
    def __call__(self, img):
        img = img.to(device)
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

def make_dataset(dir):
    num_of_cpu = joblib.cpu_count()
    delayed_funcs = [delayed(cv2.imread)(file, cv2.IMREAD_GRAYSCALE) for file in sorted(glob.glob(dir+'/*.png'))]
    parallel_pool = Parallel(n_jobs=num_of_cpu)
    images = parallel_pool(delayed_funcs)
    return images


def _shift(x):
    if x == 0:
        return 0
    if x < 0:
        return -1
    if x > 0:
        return 1

def subpixel_shift(img, shift):
    i, j = shift
    y, x = int(i), int(j)
    i, j = i - y, j - x

    img = torch.roll(img, shifts=(y, x), dims=(1, 2))

    y_shift = _shift(i)
    x_shift = _shift(j)
    i, j = abs(i), abs(j)

    img_shift = torch.roll(img, shifts=(y_shift, 0), dims=(1, 2))
    img = img * (1 - i) + img_shift * i

    img_shift = torch.roll(img, shifts=(0, x_shift), dims=(1, 2))
    img = img * (1 - j) + img_shift * j

    return img

def upsampled_dft(data, region_size, region_offset):

    im2pi = 1j * 2 * torch.pi
    shape = torch.tensor(data.shape[1:], device=device)
    dim = torch.stack([shape, region_size, region_offset], dim=1)
    dim = torch.roll(dim, (1, 0), dims=(0, 1))

    for d in dim:
        n, u, a = d.tolist()
        kernel = (torch.arange(u) - a)[:, None]
        fft_req = torch.fft.fftfreq(int(n), 10., dtype=torch.float64)
        kernel = kernel * fft_req
        kernel = torch.exp(-im2pi * kernel).to(device)
        data = torch.tensordot(kernel, data, dims=([1], [-1]))

    return data


def phase_cross_correlation(img1, img2):

    fft1 = torch.fft.rfftn(img1)
    fft2 = torch.fft.rfftn(img2)

    shape = fft1.shape[1:]
    R = (fft1 * fft2.conj()).type(torch.complex128)
    corr = torch.fft.ifftn(R)

    h, w = corr.shape[1:]
    ind = torch.argmax(torch.abs(corr))
    shift = torch.stack([ind // w, ind % w]).type(torch.float32)

    midpoints = torch.fix(torch.tensor(shape, dtype=torch.float32, device=device) / 2)

    shift[shift > midpoints] -= torch.tensor(shape, device=device)[shift > midpoints]

    region_size = torch.tensor([15., 15.], device=device)
    dftshift = torch.tensor([7., 7.], device=device)

    region_offset = dftshift - shift * 10

    corr = upsampled_dft(R.conj(), region_size, region_offset).conj()

    h, w = corr.shape[:-1]
    ind = torch.argmax(torch.abs(corr))
    shift_sub = torch.stack([ind // w, ind % w]).type(torch.float32)
    shift_sub -= dftshift
    shift = shift + shift_sub / 10

    return shift

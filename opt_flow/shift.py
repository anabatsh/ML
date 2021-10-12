import torch
import cv2

torch.pi = torch.acos(torch.zeros(1)).item() * 2

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


def corr_clarification(R, shift, device, up_c=100):

    im2pi = 1j * 2 * torch.pi

    up_c_part = int(up_c / 2)
    dfshift = up_c_part - shift * up_c
    shape = R.shape[1:]

    for i in (1, 0):
        kernel = torch.arange(1.5 * up_c).to(device)[:, None] - dfshift[i]
        fft_req = torch.fft.fftfreq(shape[i], up_c, dtype=torch.float64, device=device)
        kernel = kernel * fft_req
        kernel = torch.exp(-im2pi * kernel).to(device)
        R = torch.tensordot(kernel, R, dims=([1], [-1]))

    R = R.transpose(2, 0).transpose(1, 2)
    sub_shift =  unravel_ind(R.conj()).to(device) - up_c_part
    sub_shift = sub_shift / up_c

    return sub_shift


def unravel_ind(corr):
    w = corr.shape[-1]
    max_ind = torch.argmax(torch.abs(corr))
    shift = torch.tensor([max_ind // w, max_ind % w])
    return shift


def phase_cross_correlation(base_img, mov_img, device, up_c=100):

    base_fft = torch.fft.fftn(base_img)
    mov_fft = torch.fft.fftn(mov_img)

    R = (base_fft * mov_fft.conj()).type(torch.complex128)
    corr = torch.fft.ifftn(R)

    shape = corr.shape[1:]
    shape = torch.tensor(shape, device=device)
    shift =  unravel_ind(corr).to(device)
    midpoints = torch.fix(shape / 2)
    shift[shift > midpoints] -= shape[shift > midpoints]

    sub_shift = corr_clarification(R.conj(), shift, device, up_c=up_c)

    shift = shift + sub_shift

    return shift



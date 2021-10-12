import numpy as np

#torch
import torch
import torch.nn as nn
from torch import optim

#display
import matplotlib.pyplot as plt
from IPython.display import clear_output
from skimage.morphology import binary_dilation, binary_erosion, area_opening, square, disk, area_closing

from skimage.io import imread
from skimage.util import invert, img_as_float
from skimage.filters import threshold_otsu, median
from skimage.morphology import binary_dilation, binary_erosion, area_opening, square, disk
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.color import label2rgb, rgb2gray
from skimage.transform import rotate
from skimage.feature import blob_dog

from scipy import ndimage as ndi

def _get_coords(h, w, k):

    x = np.arange(w // k) * k
    y = np.arange(h // k) * k
    xy = np.stack(np.meshgrid(x, y)[::-1], axis=-1)
    xy = xy.reshape(-1, 2)
    return xy

def _sub_segmentation(net, sub_img):

    net.eval()
    sub_img = sub_img.unsqueeze(0)

    with torch.no_grad():
        output = net(sub_img)
        pred = (torch.sigmoid(output) > 0.5).type(torch.float32)
        pred = pred.squeeze(0)
        mask = pred.cpu().detach().numpy()

    return mask

def segmentation(net, img, s, overlap, tr):

    h, w, c = img.shape
    k = s - overlap
    h_new = h + k - h % k + overlap
    w_new = w + k - w % k + overlap

    img_new = np.zeros((h_new+1, w_new+1, c), dtype=np.float32)
    img_new[:h,:w] = img

    mask = np.zeros((h_new, w_new, 1))

    for (y, x) in _get_coords(h_new, w_new, k):
        sub_img = tr(img_new[y:y+s, x:x+s,...])
        mask[y:y+s, x:x+s] = np.transpose(_sub_segmentation(net, sub_img), (1, 2, 0))

    mask = mask[:h, :w]
    mask[mask > 0] = 1

    mask_new = np.transpose(mask, (2, 0, 1))

    mask_new[0] = binary_dilation(mask_new[0], disk(11))
    mask_new[0] = binary_erosion(mask_new[0], disk(11))
    mask_new[0] = binary_erosion(mask_new[0], disk(3))
    mask_new[0] = binary_dilation(mask_new[0], disk(3))


    mask_new[0] = area_opening(mask_new[0], 1500)
    mask_new[0] = area_closing(mask_new[0], 600)
    mask_new = np.transpose(mask_new, (1, 2, 0))

    mask = mask_new
    return mask

def train(net, trainloader, opt, device,
          criterion, num_epochs=2, verbose_num_iters=10):

    loss_trace = []
    correct_trace = []

    iter_i = 0
    for epoch_i in range(num_epochs):
        for batch in trainloader:
            imgs, masks = batch['img'], batch['mask']
            imgs = imgs.to(device)
            masks = masks.to(device)

            outputs = net(imgs).to(device)

            opt.zero_grad()

            loss = criterion(outputs, masks)
            preds = (torch.sigmoid(outputs) > 0.5).type(torch.float32)

            correct = (preds==masks).type(torch.float32).mean()

            loss.backward()
            opt.step()

            loss_trace.append((iter_i, loss.item()))
            correct_trace.append((iter_i, correct.item()))

            iter_i += 1

            if iter_i % verbose_num_iters == 0:
                clear_output(wait=True)
                plt.figure(figsize=(22, 5))

                plt.subplot(1, 4, 1)
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
                plt.plot([p[0] for p in loss_trace],
                         [p[1] for p in loss_trace])

                plt.subplot(1, 4, 2)
                plt.xlabel('Iteration')
                plt.ylabel('Accuracy')
                plt.plot([p[0] for p in correct_trace],
                         [p[1] for p in correct_trace], color='orange')

                with torch.no_grad():
                    mask = masks[0].cpu().detach().numpy()[0]
                    pred = preds[0].cpu().detach().numpy()[0]

                    plt.subplot(1, 4, 3)
                    plt.imshow(mask)
                    plt.title('true mask')
                    plt.axis('off')

                    plt.subplot(1, 4, 4)
                    plt.imshow(pred)
                    plt.title('predict mask')
                    plt.axis('off')

                plt.show()

    return loss_trace[-1][1], correct_trace[-1][1]

def show_segmentation(net, img, tr):
    mask = segmentation(net, img, 200, 10, tr)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title('img')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(mask)
    plt.title('mask')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow((mask*img).astype(np.uint8))
    plt.title('img_mask')
    plt.axis('off')

    plt.show()


def find_pizza(net, img, s, overlap, tr):

    pizzeria = []
    mask = segmentation(net, img, s, overlap, tr)
    label_image = label(mask)

    for region in regionprops(label_image):

        pizza = []
        # ограничивающая рамка триминошки
        min_y, min_x, _, max_y, max_x, _ = region.bbox

        # оригинальное изображение внутри этой рамки
        sub_img = img[min_y:max_y, min_x:max_x]

        # центр триминошки
        c_y, c_x, _ = region.centroid
        c_y, c_x = int(np.fix(c_y)), int(np.fix(c_x))
        pizza.append((c_y, c_x))

        # отступы центра триминошки
        l, r = c_x - min_x, max_x - c_x
        u, d = c_y - min_y, max_y - c_y

        # радиус черного квадрата trimino_mask, в который будет вписана триминошка так,
        # чтобы центр триминошки совпадал с центром квадрата
        rad = max(l, r, u, d) + 10

        trimino_mask = np.zeros((2*rad, 2*rad, 1))

        trimino_mask[rad-u:d-rad, rad-l:r-rad] = region.image

        # распределение расстояний от цетра триминошки
        x, y, c = np.indices((2*rad+1, 2*rad+1, 1))
        disk_ = (x - rad)**2 + (y - rad)**2 < rad**2
        disk_ = ndi.distance_transform_edt(disk_)
        disk_ = disk_[:-1, :-1]

        trimino_dist = disk_ * trimino_mask

        # определение наиболее удаленных точек (углов триминошки)
        tr_d_2d = trimino_dist.transpose(2, 0, 1)[0]
        tr_d_2d[tr_d_2d == tr_d_2d.min()] = tr_d_2d.max()
        tr_d_2d = -tr_d_2d < -16.

        # для каждого угла определяется ближайшая область в триминошке
        corners = label(1-tr_d_2d)
        for corner in regionprops(corners):
            # координата угла
            c_c_x, c_c_y = corner.centroid
            c_c_x, c_c_y = int(np.fix(c_c_x)), int(np.fix(c_c_y))

            # размер области
            # радиус равен расстоянию от координаты угла до центра триминошки
            region_rad = (rad - c_c_x)**2 + (rad - c_c_y)**2

            x, y, c = np.indices(trimino_mask.shape)

            x = x - c_c_x
            y = y - c_c_y

            # вычисление сектора круга с углом 60
            disk_new = x**2 + y**2 < region_rad - 300

            norm = np.sqrt((x**2 + y**2) * region_rad) + 1e-04

            cosin = np.divide(x * (rad - c_c_x) + y * (rad - c_c_y), norm)
            sector = np.arccos(cosin) < 0.52
            sector = sector * disk_new

            # определение ближайшей к углу области триминошки
            corner_region = sector * trimino_mask

            # обрезка до нужного размера (sub_img)
            corner_region = corner_region[rad-u:d-rad, rad-l:r-rad]
            corner_region = (corner_region*sub_img).astype(np.int)

            sub_pizza = corner_region
            h_s, w_s = sub_pizza.shape[:2]
            res = np.zeros((h_s, w_s))

            for j in range(3):

                R = sub_pizza[...,j].copy()
                mask = sub_pizza[...,0] > 0
                f = R[mask].mean()
                R[~mask] = f

                R_0 = median(R > (f + 26), disk(3))
                if j == 1:
                    R_1 = median(R < (f - 18), disk(3))
                else:
                    R_1 = median(R < (f - 25), disk(3))

                res += R_0
                res += R_1

            blobs = blob_dog(res, threshold=0.1, min_sigma=1.0, max_sigma=4)

            pepperoni = 0
            for blob in blobs:
                if blob[0] > 5.0 and blob[1] > 5.0:
                    pepperoni += 1

            pizza.append(pepperoni)

        pizzeria.append(pizza)

    return pizzeria


def print_classifications(net, imgs, names, tr, text_size=7, x_shift=0, y_shift=0):
    for (img, name) in zip(imgs, names):
        with open('results/'+name+'.ans', 'w') as f:
            pizzeria = find_pizza(net, img, 200, 10, tr)
            f.write(f'{len(pizzeria)}\n')
            for pizza in pizzeria:
                if len(pizza) < 3:
                    continue
                if len(pizza) == 3:
                    pizza.append(0)
                (p_x, p_y), p_1, p_2, p_3 = pizza[:4]
                if p_1 == 6:
                    if p_2 * p_3 == 0:
                        p_1 = 5
                    else:
                        p_1 = 4

                if p_2 == 6:
                    if p_1 * p_3 == 0:
                        p_2 = 5
                    else:
                        p_2 = 4
                if p_3 == 6:
                    if p_1 * p_2 == 0:
                        p_3 = 5
                    else:
                        p_3 = 4
                if p_x < 100:
                    s = ' '
                else:
                    s = ''
                f.write(f'{p_x}, {p_y}; {p_1}, {p_2}, {p_3}\n')


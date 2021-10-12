import numpy as np

#torch
import torch
import torch.nn as nn
from torch import optim
from IPython.display import clear_output

#display
import matplotlib.pyplot as plt

from skimage.morphology import binary_dilation, binary_erosion, area_opening, disk
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops, perimeter
from skimage.color import label2rgb, rgb2gray


from sklearn.preprocessing import normalize, StandardScaler
from scipy.spatial import distance
import math

import joblib
from joblib import Parallel, delayed

import utils
from sklearn.preprocessing import normalize, StandardScaler
from scipy.spatial import distance
import networkx as nx


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
                    plt.imshow(mask, cmap='gray')
                    plt.title('true mask')
                    plt.axis('off')

                    plt.subplot(1, 4, 4)
                    plt.imshow(pred, cmap='gray')
                    plt.title('predict mask')
                    plt.axis('off')

                plt.show()

    return loss_trace[-1][1], correct_trace[-1][1]

def segmentation(net, img, tr):

    net.eval()
    
    img = utils._resize(img)
    img = tr(img).unsqueeze(0)

    with torch.no_grad():
        output = net(img)
        pred = (torch.sigmoid(output) > 0.5).type(torch.float32)
        pred = pred.squeeze(0)
        mask = pred.cpu().detach()[0].numpy()
        
    mask = utils._deresize(mask)
    mask = mask == 1.
    mask = area_opening(mask, 1e+4)
    
    return mask

def segmentation_list(images, model, tr):
    masks = [segmentation(model, img[...,:3], tr) for img in images]
    return masks

def fingers_segmentation(mask):
    
    mask_new = binary_erosion(mask, disk(55))
    mask_new = binary_dilation(mask_new, disk(55))
    
    fingers = mask.copy()
    fingers[mask_new] = 0

    fingers = binary_erosion(fingers, disk(5))
    fingers = binary_dilation(fingers, disk(5))
    
    fingers = area_opening(fingers, 1000)

    return fingers
    
    
def fingers_length(fingers, img, mask):
    
    f_length = []
    f_width = []
    f_line = []
    
    regions = regionprops(label(fingers))
    for region in regions:
        area = region.area
        if area > 1e+3:
            if area > 1e+4:
                
                minr, minc, maxr, maxc = region.bbox
                sep_fing = img[minr:maxr, minc:maxc] 
                
                sep_fing = rgb2gray(sep_fing) * region.image
                sep_fing = sep_fing > 0.2

                mask[minr:maxr, minc:maxc] = np.where(region.image, 
                                                      sep_fing, 
                                                      mask[minr:maxr, minc:maxc])
                mask_copy = np.zeros_like(mask)
                mask_copy[minr:maxr, minc:maxc] = sep_fing
                                
                label_new_region = label(mask_copy)
                new_regions = regionprops(label_new_region)
                regions.extend(new_regions)
                
            else:
                
                c_y, c_x = region.centroid
                orientation = region.orientation

                rad_a = region.minor_axis_length
                rad_b = region.major_axis_length * 0.8

                angle_a = math.cos(orientation)
                angle_b = math.sin(orientation)

                x1 = c_x + angle_b * 0.5 * rad_b
                y1 = c_y + angle_a * 0.5 * rad_b

                x2 = c_x - angle_b * 0.5 * rad_b
                y2 = c_y - angle_a * 0.5 * rad_b
                
                x3 = c_x + angle_a * 0.5 * rad_a
                y3 = c_y - angle_b * 0.5 * rad_a

                x4 = c_x - angle_a * 0.5 * rad_a
                y4 = c_y + angle_b * 0.5 * rad_a

                
                f_length.append(round(rad_b, 2)) 
                f_width.append(round(rad_a, 2)) 
                f_line.append([(x1, x2), (y1, y2), (x3, x4), (y3, y4)])
                
    return f_length, f_width, f_line
    
    
def img_coder(img, mask, fingers=None):
    
    if fingers is None:
        fingers = fingers_segmentation(mask)
    
    code = {}
    
    f_length, f_width, f_line = fingers_length(fingers, img, mask)
    f_length = np.array(f_length)
    f_width = np.array(f_width)

    indices = np.argsort(f_length)
    f_length = np.take_along_axis(f_length, indices, axis=0)
    f_width = np.take_along_axis(f_width, indices, axis=0)

    for i, (l, w) in enumerate(zip(f_length, f_width)):
        code[f'l_{i+1}'] = l
        code[f'w_{i+1}'] = w

    code['s'] = mask.sum()
    code['p'] = round(perimeter(mask), 2)
    
    names = ('r', 'g', 'b')
    for i in range(3):
        C = img[...,i] * mask
        C_hist, bins = np.histogram(C, bins=7, range=(0.15, 0.85))
        C_hist = C_hist / C_hist.sum()
        for c, b in zip(C_hist, bins):
            code[f'{names[i]}_{round(b+0.05, 1)}'] = round(c, 3)
            
    return code, f_line
    
def coding_list(images, masks):
    
    num_of_cpu = joblib.cpu_count()

    delayed_funcs = [delayed(img_coder)(img, mask) for img, mask in zip(images, masks)]
    parallel_pool = Parallel(n_jobs=num_of_cpu)
    codes_lines = parallel_pool(delayed_funcs)

    codes = [code_line[0] for code_line in codes_lines]
    lines = [code_line[1] for code_line in codes_lines]

    return codes, lines
    

def _standard(df):
    X = df.to_numpy()
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = normalize(X)
    return X

def _dist(X):
    return distance.cdist(X, X, 'euclidean')

def k_neighbors(df, images_names, k=3):
    X = _standard(df)
    dist = _dist(X)
    dist_sort = np.argsort(dist, axis=1)[:,:k+1]
    return dist_sort, np.take(images_names, dist_sort)
    
def clustering(neighbors_3):
    
    clusters = []
    G = nx.Graph()

    for i, j in neighbors_3[:,:2]:
        G.add_edge(i, j)

    for g in nx.connected_components(G):
        clusters_img = [idx for idx in g]
        clusters.append(clusters_img)
        
    return clusters



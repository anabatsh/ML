import numpy as np

#display
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

import utils
import net_process
from IPython.display import display_html
from itertools import chain,cycle
from skimage.segmentation import clear_border
from skimage.morphology import area_opening
from skimage.morphology import convex_hull_image



import pandas as pd

def show_images(images, images_names):
    
    plt.figure(figsize=(16, 4))
    plt.suptitle('Исходные изображения')
    for i in range(4):
        plt.subplot(1, 4, i+1)

        plt.imshow(images[i])
        plt.title(images_names[i])
        plt.axis('off')

def show_making_mask(img):
    
    plt.figure(figsize=(16, 5))
    plt.suptitle('Процесс вычисления маски')

    img = rgb2gray(img)
    palm = img > img.mean()
    plt.subplot(1, 5, 1)
    plt.imshow(palm, cmap='gray')
    plt.title('Бинаризация 1 \n(по среднему)')
    plt.axis('off')
    
    thresh = np.quantile(img, 0.8)
    bin_img = (img > thresh)
    plt.subplot(1, 5, 2)
    plt.imshow(bin_img, cmap='gray')
    plt.title('Бинаризация 2 \n(по границе 0.85)')
    plt.axis('off')
    
    bin_img = clear_border(bin_img)
    bin_img = area_opening(bin_img, 300)
    plt.subplot(1, 5, 3)
    plt.title('Очищенная \nбинаризация 2')
    plt.imshow(bin_img, cmap='gray')
    plt.axis('off')
    
    bin_img = convex_hull_image(bin_img)
    plt.subplot(1, 5, 4)
    plt.title('Выпуклая оболочка \nбинаризации 2')
    plt.imshow(bin_img, cmap='gray')
    plt.axis('off')
    
    bin_img = bin_img * palm
    plt.subplot(1, 5, 5)
    plt.title('Совмещение \n1 и 2 бинаризаций')
    plt.imshow(bin_img, cmap='gray')
    plt.axis('off')
    
def show_artificial_mask(images, images_names):

    plt.figure(figsize=(16, 4))
    plt.suptitle('Искусственные маски')
    
    for i, img in enumerate(images):
        plt.subplot(2, 5, i+1)
        img = utils._resize(img)
        mask = utils._make_mask(img)
        mask_3 = np.stack((mask,) * 3, axis=-1)
        img_to_show = np.hstack((img, mask_3))
        plt.imshow(img_to_show)
        plt.title(images_names[i])
        plt.axis('off')
    
    
def show_dataset_samples(dataset):

    plt.figure(figsize=(16, 8))
    plt.suptitle('Выборка сэмплов из обучающего датасета')

    indices = np.arange(200, step=10, dtype=np.int32)

    for i, idx in enumerate(indices):
        plt.subplot(4, 5, i+1)
        sample = dataset[idx]
        img, mask = sample['img'], sample['mask']
        img = img.cpu().transpose(0, 1).transpose(1, 2).numpy()
        img = (img + 1) / 2
        mask = mask.cpu().numpy()[0]
        mask_3 = np.stack((mask,) * 3, axis=-1)
        img_to_show = np.hstack((img, mask_3))
        plt.imshow(img_to_show)
        plt.axis('off')
        
def show_segmentation_results(images, masks):

    plt.figure(figsize=(16, 4))
    plt.suptitle('Результаты сегментации')
    
    for i, (img, mask) in enumerate(zip(images, masks)):
        plt.subplot(2, 5, i+1)
        mask_3 = np.stack((mask,) * 3, axis=-1)
        img_to_show = np.hstack((img[...,:3], mask_3))
        plt.imshow(img_to_show)
        plt.axis('off')
        
        
def show_separated_fingers(img, mask, a, b, c, d):
    
    plt.figure(figsize=(16, 4))
    plt.suptitle('Разделение склеенных пальцев')
    
    mask = mask.copy()
    plt.subplot(1, 5, 1)
    plt.imshow(mask, cmap='gray')
    plt.title('original mask')
    plt.axis('off')

    glued_fingers = mask[a:b, c:d]
    plt.subplot(1, 5, 2)
    plt.imshow(glued_fingers, cmap='gray')
    plt.title('glued fingers')
    plt.axis('off')

    sep_fing = rgb2gray(img[a:b, c:d])
    plt.subplot(1, 5, 3)
    plt.imshow(sep_fing, cmap='gray')
    plt.title('original image')
    plt.axis('off')

    sep_fing = sep_fing > 0.2
    plt.subplot(1, 5, 4)
    plt.imshow(sep_fing, cmap='gray')
    plt.title('segmented fingers')
    plt.axis('off')

    mask[a:b, c:d] = np.where(sep_fing, sep_fing, 0)
    plt.subplot(1, 5, 5)
    plt.imshow(mask, cmap='gray')
    plt.title('final mask')
    plt.axis('off')

    plt.show()
    
    
def show_colors_hist(img, mask):
    
    plt.figure(figsize=(16, 3))
    plt.suptitle('Распределение интенсивности цвета по каждому каналу')
    plt.subplot(1, 4, 1)
    plt.imshow(img)
    plt.axis('off')
    
    c = ('red', 'green', 'blue')
    for i in range(3):
        plt.subplot(1, 4, i+2)

        C = img[...,i] * mask
        bins = np.arange(0.15, 0.95, step=0.1)
        plt.hist(C[C > 0], bins=bins, color=c[i])
        plt.xlim(0.15, 0.85)
        plt.yticks(())
        plt.ylabel(c[i])
    plt.show()
    
def show_fingers_processing(img, model, tr):
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 4))
    plt.suptitle('Обработка ладони: вычисление длины и ширины пальцев')

    ax1.imshow(img)
    ax1.axis('off')
    ax1.set_title('img')
    
    mask = net_process.segmentation(model, img[...,:3], tr)
    ax2.imshow(mask, cmap='gray')
    ax2.axis('off')
    ax2.set_title('mask')
    
    fingers = net_process.fingers_segmentation(mask)
    ax3.imshow(fingers, cmap='gray')
    ax3.axis('off')
    ax3.set_title('fingers')
    
    code, f_line = net_process.img_coder(img, mask, fingers)
    
    ax4.imshow(mask, cmap='gray')
    ax4.axis('off')

    for (x1, x2), (y1, y2), (x3, x4), (y3, y4) in f_line:
        ax4.plot((x1, x2), (y1, y2), '-r', linewidth=2.5)
        ax4.plot((x3, x4), (y3, y4), '-r', linewidth=2.5)

    ax4.set_title('fingers lines')
    plt.show()
    
    
def show_fingers_processing_results(masks, lines):
    
    plt.figure(figsize=(16, 4))
    plt.suptitle('Результаты обработки ладоней')
    
    for i in range(len(masks)):
        plt.subplot(1, 5, i+1)

        plt.imshow(masks[i], cmap='gray')
        plt.axis('off')

        for (x1, x2), (y1, y2), (x3, x4), (y3, y4) in lines[i]:
            plt.plot((x1, x2), (y1, y2), '-r', linewidth=2.5)
            plt.plot((x3, x4), (y3, y4), '-r', linewidth=2.5)
    plt.show()
    
    
    
def show_tables(*args,titles=cycle([''])):
    html_str=''
    for i, (df, title) in enumerate(zip(args, chain(titles,cycle(['</br>'])) )):
        df = df.copy()
        df['name'] = [f'_span_left_{x}_span_right_' for x in df['name']]
        html_str += '<th style="text-align:center"><td style="vertical-align:top">'
        html_str += f'<h2 style="text-align:center">{title}</h2>'
        html_str += df.to_html(index=False).replace('table','table style="display:inline"')
        html_str += '</td></th>'
        
        html_str = html_str.replace('_span_left_', '<span style="font-weight:bold">')
        html_str = html_str.replace('_span_right_', '</span>')
        
    display_html(html_str,raw=True)
    
    
    
def show_3_neigbors_table(neighbors_3):
    df_neighbors_3_list = [pd.DataFrame(neighbors_3[i*20: min((i+1)*20, len(neighbors_3))], 
                                       columns=['name', 'n_1', 'n_2', 'n_3']) for i in range(5)]

    show_tables(*df_neighbors_3_list, titles=['0 - 20', '20 - 40', 
                                              '40 - 60', '60 - 80', '80 - 100'])
                                                       
                                                       
                                                       
def show_3_neigbors_images(neighbors_3, images, images_names):
    for neig in neighbors_3:
        plt.figure(figsize=(16, 4))

        for i, n in enumerate(neig):
            plt.subplot(1, 4, i+1)
            name = images_names[n]
            title = f'img : {name}' if i == 0 else f'n_{i} : {name}'
            plt.title(title)
            plt.imshow(images[n])
            plt.axis('off')
        plt.show()
        
        
def show_clusters_table(clusters):
    
    clusters_for_table = []
    
    for i, cluster in enumerate(clusters):
        l = len(cluster) 
        clusters_for_table.append([i] + cluster)
        if l < 7:
            clusters_for_table[i] += [-1] * (7 - l)
            
    df_clusters_list = [pd.DataFrame(clusters_for_table[i*9: (i+1)*9], 
                                     columns=['name'] + [''] * 7).replace({-1: ''}) for i in range(3)]

    show_tables(*df_clusters_list, titles=['0 - 9', '10 - 19', '20 - 27'])

    
def show_clusters_images(clusters, images, images_names):
    
    for i, cluster in enumerate(clusters):

        plt.figure(figsize=(16, 5))

        for i, idx in enumerate(cluster):
            plt.subplot(1, 7, i+1)
            plt.title(f'img: {idx}')
            plt.imshow(images[images_names.index(idx)])
            plt.axis('off')

        plt.show()

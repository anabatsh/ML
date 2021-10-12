import joblib
from joblib import Parallel, delayed

import zipfile
import glob
import natsort

import os
import cv2

def make_dataset(dir):
    num_of_cpu = joblib.cpu_count()
    delayed_funcs = [delayed(cv2.imread)(file, cv2.IMREAD_GRAYSCALE) for file in sorted(glob.glob(dir+'/*'))]
    parallel_pool = Parallel(n_jobs=num_of_cpu)
    images = parallel_pool(delayed_funcs)
    return images

def dataset(dir):
    zip_file = dir
    z = zipfile.ZipFile(zip_file, 'r')
    z.extractall()
    return make_dataset(dir[:-4])

#!/usr/bin/env python

import chainer
from chainer.dataset import dataset_mixin

from PIL import Image
import numpy as np

import os
import six
import random


class PreprocessedDataset(dataset_mixin.DatasetMixin):

    def __init__(self, paths, crop_size=None, dtype=np.float32):
        if isinstance(paths, six.string_types):
            with open(paths) as paths_file:
                paths = [path.strip() for path in paths_file]
        self._paths = paths
        #self._root = root
        self.crop_size = crop_size
        self._dtype = dtype
        

    def __len__(self):
        return len(self._paths)

    def get_example(self, i):
        #path = os.path.join(self._root, self._paths[i])
        path = self._paths[i]
        img = Image.open(path, 'r')

        # Converting format of chainer (np.ndarray, float32, CHW)
        array = np.asarray(img, dtype=np.float32)
        if array.ndim == 2:
            array = array[:, :, np.newaxis] # image is greyscale
        x = array.transpose(2, 0, 1)

        # crop if crop_size is designated
        if not self.crop_size is None:
            _,h,w = x.shape
            top = random.randint(0, h - self.crop_size)
            left = random.randint(0, w - self.crop_size)
            bottom = top + self.crop_size
            right = left + self.crop_size
            x = x[:, top:bottom, left:right]
        
        # Nomilizing in [-1, 1]
        x = (x - 128.0) / 128.0 
    
        # Random horizontal flipping
        if np.random.randint(2):
            x = x[:,:,::-1]

        return x


# メモリの有効活用のため、事前にリサイズを行っておく関数
def resize_data(paths, resize, cashe_dir='./cashe'):

    if isinstance(paths, six.string_types):
        with open(paths) as paths_file:
            paths = [path.strip() for path in paths_file]

    if not os.path.exists(cashe_dir):
        os.mkdir(cashe_dir)

    output_paths = []
    N = len(paths)
    for i, path in enumerate(paths):
        print('{}/{}'.format(i+1, N))
        #img_file = os.path.join(root, path)
        #print(img_file)
        
        path_split = path.split('/')
        output_dir = os.path.join(cashe_dir, path_split[-2])
        output_file = os.path.join(output_dir, path_split[-1])
        output_paths.append(output_file)

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        if not os.path.exists(output_file):
            img = Image.open(path, 'r')
            # Crop center
            if img.size[0] > img.size[1]:
                sub = (img.size[0] - img.size[1]) // 2
                img = img.crop((sub, 0, img.size[0]-1-sub, img.size[1]-1))
            elif img.size[0] < img.size[1]:
                sub = (img.size[1] - img.size[0]) // 2
                img = img.crop((0, sub, img.size[0]-1, img.size[1]-1-sub))
            #print(img.size)
            # Resize
            img = img.resize((resize, resize), Image.ANTIALIAS) # BICUBIC is not good for downscaling
            img.save(output_file)

    return output_paths

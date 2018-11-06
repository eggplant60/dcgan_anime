#!/usr/bin/env python

import numpy
import math

import chainer
from chainer import cuda
from chainer.utils import type_check
from chainer import function

import chainer.functions as F
import chainer.links as L


def add_noise(h, sigma=0.2):
    xp = cuda.get_array_module(h.data)
    if chainer.config.train:
        return h + sigma * xp.random.randn(*h.shape)
    else:
        return h

    
class Generator(chainer.Chain):
    def __init__(self, n_hidden, wscale=0.02, z_sampling='gaussian'):
        self.n_hidden = n_hidden
        self.wscale = wscale
        self.z_sampling = z_sampling

        super(Generator, self).__init__()
        self.n_hidden = n_hidden

        with self.init_scope():
            w = chainer.initializers.Normal(self.wscale)
            self.l0z = L.Linear(self.n_hidden, 6*6*512, initialW=w)
            self.dc1 = L.Deconvolution2D(512, 256, 4, stride=2, pad=1, initialW=w)
            self.dc2 = L.Deconvolution2D(256, 128, 4, stride=2, pad=1, initialW=w)
            self.dc3 = L.Deconvolution2D(128, 64, 4, stride=2, pad=1, initialW=w)
            self.dc4 = L.Deconvolution2D(64, 3, 4, stride=2, pad=1, initialW=w)
            self.bn0l = L.BatchNormalization(6*6*512)
            self.bn1 = L.BatchNormalization(256)
            self.bn2 = L.BatchNormalization(128)
            self.bn3 = L.BatchNormalization(64)
            
    def make_hidden(self, batchsize):
        size = (batchsize, self.n_hidden, 1, 1)
        if self.z_sampling == 'gaussian':
            #print('aaa')
            return numpy.random.normal(loc=0.0, scale=1.0, size=size).astype(numpy.float32)
        else:
            return numpy.random.uniform(-1, 1, size=size).astype(numpy.float32)
            
    def __call__(self, z):
        h = F.reshape(F.leaky_relu(self.bn0l(self.l0z(z))),
                      (z.data.shape[0], 512, 6, 6))
        #print('G: {}'.format(h.shape))
        h = F.leaky_relu(self.bn1(self.dc1(h)))
        h = F.leaky_relu(self.bn2(self.dc2(h)))
        h = F.leaky_relu(self.bn3(self.dc3(h)))
        # 最終層にはBN入れない
        # https://elix-tech.github.io/ja/2017/02/06/gan.html
        x = F.tanh(self.dc4(h)) # [-1, 1]
        return x

    
# for WGAN-GP
class MyLayerNormalization(L.LayerNormalization):
    def __init__(self, ch):
        super(MyLayerNormalization, self).__init__()
    
    def __call__(self, x):
        shape = x.shape
        x = x.reshape((shape[0], -1)) # flatten
        x = super().__call__(x)
        x = x.reshape(shape)
        return x

    
class Discriminator(chainer.Chain):
    def __init__(self, init_sigma=0.2, wscale=0.02):
        self.sigma = init_sigma
        self.wscale = wscale

        super(Discriminator, self).__init__()
        with self.init_scope():
            w = chainer.initializers.Normal(self.wscale)
            self.c0 = L.Convolution2D(3, 64, 4, stride=2, pad=1, initialW=w)
            self.c1 = L.Convolution2D(64, 128, 4, stride=2, pad=1, initialW=w)
            self.c2 = L.Convolution2D(128, 256, 4, stride=2, pad=1, initialW=w)
            self.c3 = L.Convolution2D(256, 512, 4, stride=2, pad=1, initialW=w)
            self.l4l = L.Linear(6*6*512, 1, initialW=w) # bottom_width = 6
            self.bn1 = L.BatchNormalization(128)
            self.bn2 = L.BatchNormalization(256)
            self.bn3 = L.BatchNormalization(512)
            # self.bn1 = MyLayerNormalization(None)
            # self.bn2 = MyLayerNormalization(None)
            # self.bn3 = MyLayerNormalization(None)
        
    def __call__(self, x):
        h = add_noise(x,self.sigma)
        # 入力層にはBN入れない
        # https://elix-tech.github.io/ja/2017/02/06/gan.html
        h = F.leaky_relu(add_noise(self.c0(h),self.sigma)) # no bn because images from generator will katayotteru?
        h = F.leaky_relu(add_noise(self.bn1(self.c1(h)),self.sigma))
        h = F.leaky_relu(add_noise(self.bn2(self.c2(h)),self.sigma))
        h = F.leaky_relu(add_noise(self.bn3(self.c3(h)),self.sigma))
        y = self.l4l(h)
        return y

    
    def shift_sigma(self, alpha_sigma=0.9):
        self.sigma *= alpha_sigma
        return self.sigma

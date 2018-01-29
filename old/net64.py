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

    
# def elu(x, alpha=1.0):
#     """Exponential Linear Unit function."""
#     # https://github.com/muupan/chainer-elu
#     return ELU(alpha=alpha)(x)


class Generator(chainer.Chain):
    def __init__(self, n_hidden):
        self.n_hidden = n_hidden

        super(Generator, self).__init__()
        self.n_hidden = n_hidden

        with self.init_scope():
            w0 = chainer.initializers.Normal(0.02*math.sqrt(self.n_hidden))
            w1 = chainer.initializers.Normal(0.02*math.sqrt(4*4*512))
            w2 = chainer.initializers.Normal(0.02*math.sqrt(4*4*256))
            w3 = chainer.initializers.Normal(0.02*math.sqrt(4*4*128))
            w4 = chainer.initializers.Normal(0.02*math.sqrt(4*4*64))                                
            self.l0z = L.Linear(self.n_hidden, 4*4*512, initialW=w0)
            self.dc1 = L.Deconvolution2D(512, 256, 4, stride=2, pad=1, initialW=w1)
            self.dc2 = L.Deconvolution2D(256, 128, 4, stride=2, pad=1, initialW=w2)
            self.dc3 = L.Deconvolution2D(128, 64, 4, stride=2, pad=1, initialW=w3)
            self.dc4 = L.Deconvolution2D(64, 3, 4, stride=2, pad=1, initialW=w4)
            self.bn0l = L.BatchNormalization(4*4*512)
            #self.bn0 = L.BatchNormalization(512)
            self.bn1 = L.BatchNormalization(256)
            self.bn2 = L.BatchNormalization(128)
            self.bn3 = L.BatchNormalization(64)
            
    def make_hidden(self, batchsize):
        return numpy.random.uniform(-1, 1, (batchsize, self.n_hidden, 1, 1))\
            .astype(numpy.float32)
            
    def __call__(self, z):
        h = F.reshape(F.leaky_relu(self.bn0l(self.l0z(z))),
                      (z.data.shape[0], 512, 4, 4))
        #print('G: {}'.format(h.shape))
        h = F.relu(self.bn1(self.dc1(h)))
        h = F.relu(self.bn2(self.dc2(h)))
        h = F.relu(self.bn3(self.dc3(h)))
        x = F.tanh(self.dc4(h)) # [-1, 1]
        return x


class Discriminator(chainer.Chain):
    def __init__(self, sigma=0.2):
        self.sigma = sigma

        super(Discriminator, self).__init__()
        with self.init_scope():
            w0 = chainer.initializers.Normal(0.02*math.sqrt(4*4*3))
            w1 = chainer.initializers.Normal(0.02*math.sqrt(4*4*64))
            w2 = chainer.initializers.Normal(0.02*math.sqrt(4*4*128))
            w3 = chainer.initializers.Normal(0.02*math.sqrt(4*4*256))
            w4 = chainer.initializers.Normal(0.02*math.sqrt(4*4*512))
            self.c0 = L.Convolution2D(3, 64, 4, stride=2, pad=1, initialW=w0)
            self.c1 = L.Convolution2D(64, 128, 4, stride=2, pad=1, initialW=w1)
            self.c2 = L.Convolution2D(128, 256, 4, stride=2, pad=1, initialW=w2)
            self.c3 = L.Convolution2D(256, 512, 4, stride=2, pad=1, initialW=w3)
            self.l4l = L.Linear(4*4*512, 1, initialW=w4)
            #self.bn0 = L.BatchNormalization(64)
            self.bn1 = L.BatchNormalization(128)
            self.bn2 = L.BatchNormalization(256)
            self.bn3 = L.BatchNormalization(512)

        
    def __call__(self, x):
        h = add_noise(x,self.sigma)
        h = F.leaky_relu(add_noise(self.c0(h),self.sigma)) # no bn because images from generator will katayotteru?
        h = F.leaky_relu(add_noise(self.bn1(self.c1(h)),self.sigma))
        h = F.leaky_relu(add_noise(self.bn2(self.c2(h)),self.sigma))
        h = F.leaky_relu(add_noise(self.bn3(self.c3(h)),self.sigma))
        # h = F.leaky_relu(self.c0(h))    # no bn because images from generator will katayotteru?
        # h = F.leaky_relu(self.bn1(self.c1(h)))
        # h = F.leaky_relu(self.bn2(self.c2(h)))
        # h = F.leaky_relu(self.bn3(self.c3(h)))        
        y = self.l4l(h)
        return y

    
    def shift_sigma(self, co_sigma=0.9):
        self.sigma *= co_sigma
        return self.sigma
    

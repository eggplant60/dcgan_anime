#!/usr/bin/env python

from __future__ import print_function

import chainer
import chainer.functions as F
from chainer import Variable
import numpy as np

class DCGANUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop('models')
        super(DCGANUpdater, self).__init__(*args, **kwargs)

    def loss_dis(self, dis, y_fake, y_real):
        batchsize = len(y_fake)
        L1 = F.sum(F.softplus(-y_real)) / batchsize
        L2 = F.sum(F.softplus(y_fake)) / batchsize
        loss = L1 + L2
        chainer.report({'loss': loss}, dis)
        return loss

    def loss_gen(self, gen, y_fake):
        batchsize = len(y_fake)
        loss = F.sum(F.softplus(-y_fake)) / batchsize
        chainer.report({'loss': loss}, gen)
        return loss

    def update_core(self):
        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('dis')

        batch = self.get_iterator('main').next()
        x_real = Variable(self.converter(batch, self.device)) # [-1, 1]
        xp = chainer.cuda.get_array_module(x_real.data)

        gen, dis = self.gen, self.dis
        batchsize = len(batch)

        y_real = dis(x_real)

        z = Variable(xp.asarray(gen.make_hidden(batchsize)))
        x_fake = gen(z)
        y_fake = dis(x_fake)

        dis_optimizer.update(self.loss_dis, dis, y_fake, y_real)
        gen_optimizer.update(self.loss_gen, gen, y_fake)


class WGANGPUpdater(chainer.training.StandardUpdater):

    def __init__(self, l=10, n_c=5, *args, **kwargs):
        self.gen, self.dis = kwargs.pop('models')
        self.l = l
        self.n_c = n_c
        super(WGANGPUpdater, self).__init__(*args, **kwargs)

    def loss_gen(self, gen, y_fake):
        batchsize = len(y_fake)
        loss = F.sum(-y_fake) / batchsize
        chainer.report({'loss': loss}, gen)
        return loss
    
    def loss_dis(self, dis, y_fake, y_real, y_hat, x_hat):
        #print(y_fake.shape)
        batchsize,_,w,h = x_hat.data.shape
        xp = dis.xp

        loss_gan = F.sum(y_fake - y_real) / batchsize / w / h

        grad, = chainer.grad([y_hat], [x_hat], enable_double_backprop=True)
        grad = F.sqrt(F.batch_l2_norm_squared(grad))
        loss_grad = self.l * F.mean_squared_error(grad, xp.ones_like(grad.data))
        
        loss = loss_gan + loss_grad
        chainer.report({'loss/gan': loss_gan, 'loss/grad': loss_grad}, dis)
        return loss

    
    def update_core(self):
        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('dis')
        gen, dis = self.gen, self.dis
        
        #xp = chainer.cuda.get_array_module(x_real.data)
        xp = gen.xp
        
        # train discriminator (critic)
        for _ in range(self.n_c):
            batch = self.get_iterator('main').next()
            x_real = Variable(self.converter(batch, self.device)) # [-1, 1]
            batchsize = len(batch)
            
            # generate
            z = Variable(xp.asarray(gen.make_hidden(batchsize)))
            x_fake = gen(z)

            # sampling along straingt lines
            e = xp.random.uniform(0., 1., (batchsize, 1, 1, 1))
            x_hat = e * x_real + (1 - e) * x_fake

            y_fake = dis(x_fake)
            y_real = dis(x_real)
            y_hat  = dis(x_hat)

            #import pdb; pdb.set_trace()
            dis_optimizer.update(self.loss_dis, dis, y_fake, y_real, y_hat, x_hat)

        # train generator
        batch = self.get_iterator('main').next()
        x_real = Variable(self.converter(batch, self.device)) # [-1, 1]
        batchsize = len(batch)
        z = Variable(xp.asarray(gen.make_hidden(batchsize)))
        x_fake = gen(z)
        y_fake = dis(x_fake)

        gen_optimizer.update(self.loss_gen, gen, y_fake)

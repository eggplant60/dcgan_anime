#!/usr/bin/env python

from __future__ import print_function
import argparse
import os
import six

import chainer
from chainer import training
from chainer.training import extensions

import cifar
from updater import DCGANUpdater, WGANGPUpdater
from visualize import out_generated_image
from dataset import *

#from chainer.datasets import TransformDataset
#from chainercv.transforms.image.resize import resize

from PIL import Image
import numpy as np

insize = 96

if insize == 32:
    from net32 import Discriminator, Generator
elif insize == 64:
    from net64 import Discriminator, Generator
elif insize == 96:
    from net96 import Discriminator, Generator
        
    
def main():
    parser = argparse.ArgumentParser(description='Chainer example: DCGAN')
    parser.add_argument('--batchsize', '-b', type=int, default=50, # default=50
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=10000, # defalt=500
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--dataset', '-i', default='train_illya.txt', # defalt=''
                        help='Directory of image files.  Default is cifar-10.')
    parser.add_argument('--out', '-o', default='result_anime_illya4', # defalt='result'
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--n_hidden', '-n', type=int, default=100,
                        help='Number of hidden units (z)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed of z at visualization stage')
    parser.add_argument('--snapshot_interval', type=int, default=1000, # defalt=1000
                        help='Interval of snapshot')
    parser.add_argument('--display_interval', type=int, default=100,
                        help='Interval of displaying log to console')
    parser.add_argument('--noise_sigma', type=float, default=0.2,     # best: 0.2
                        help='Std of noise added the descriminator')

    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# n_hidden: {}'.format(args.n_hidden))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train
    gen = Generator(n_hidden=args.n_hidden, wscale=0.02)
    dis = Discriminator(init_sigma=args.noise_sigma, wscale=0.02)

    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        gen.to_gpu()  # Copy the model to the GPU
        dis.to_gpu()

    # Setup an optimizer
    # https://elix-tech.github.io/ja/2017/02/06/gan.html
    def make_optimizer(model, alpha=0.0002, beta1=0.5): # 元論文
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(0.00001), 'hook_dec')
        return optimizer
    
    # # For WGAN
    # # Not good
    # def make_optimizer(model, alpha=0.0001, beta1=0.0, beta2=0.9):
    #     optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2)
    #     optimizer.setup(model)
    #     return optimizer

    opt_gen = make_optimizer(gen)
    opt_dis = make_optimizer(dis)

    if args.dataset == '':
        # Load the CIFAR10 dataset if args.dataset is not specified
        train, _ = cifar.get_cifar10(withlabel=False, scale=255.)
    else:
        resized_paths = resize_data(args.dataset, insize+16)
        train = PreprocessedDataset(resized_paths, crop_size=insize)
    print('{} contains {} image files'.format(args.dataset, train.__len__()))
              
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)

    # Set up a trainer
    updater = DCGANUpdater(
        models=(gen, dis),
        iterator=train_iter,
        optimizer={
            'gen': opt_gen, 'dis': opt_dis},
        device=args.gpu)
    
    # updater = WGANGPUpdater(
    #     models=(gen, dis),
    #     iterator=train_iter,
    #     optimizer={
    #         'gen': opt_gen, 'dis': opt_dis},
    #     device=args.gpu,
    #     l=10,
    #     n_c=5
    # )

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    snapshot_interval = (args.snapshot_interval, 'iteration')
    display_interval = (args.display_interval, 'iteration')
    trainer.extend(
        extensions.snapshot(filename='snapshot_iter_{.updater.iteration}.npz'),
        trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        gen, 'gen_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        dis, 'dis_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'gen/loss', 'dis/loss',
        #'epoch', 'iteration', 'gen/loss', 'dis/loss/gan', 'dis/loss/grad', # For WGAN
    ]), trigger=display_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(out_generated_image(gen, dis, 5, 5, args.seed, args.out),
                   trigger=snapshot_interval)
                   
    # 次第にDescriminatorのノイズを低減させる
    @training.make_extension(trigger=snapshot_interval)
    def shift_sigma(trainer):
        s = dis.shift_sigma(alpha_sigma=0.9) 
        print('sigma={}'.format(s))
        print('')
        
    trainer.extend(shift_sigma) # 通常のDCGANではコメントイン
    
    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()

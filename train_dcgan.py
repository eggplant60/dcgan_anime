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

#from chainer.datasets import TransformDataset
#from chainercv.transforms.image.resize import resize

from PIL import Image
import numpy as np
from chainer.dataset import dataset_mixin


insize = 96

if insize == 32:
    from net32 import Discriminator, Generator
elif insize == 64:
    from net64 import Discriminator, Generator
elif insize == 96:
    from net96 import Discriminator, Generator



class PreprocessedDataset(dataset_mixin.DatasetMixin):

    def __init__(self, paths, root='.', dtype=np.float32):
        if isinstance(paths, six.string_types):
            with open(paths) as paths_file:
                paths = [path.strip() for path in paths_file]
        self._paths = paths
        self._root = root
        self._dtype = dtype

    def __len__(self):
        return len(self._paths)

    def get_example(self, i):
        path = os.path.join(self._root, self._paths[i])
        img = Image.open(path, 'r')

        # # 以下は逐次処理だと間に合わないので前処理に移行
        # 3. Random cropping
        # ! PIL.Image.Image 形式なので以下の演算はエラー？
        # -> img.crop()を使えばよい
        #_, h, w = img.shape
        #top = np.random.randint(0, h//3)
        #left = np.random.randint(0, w//3)
        #img = img[:, top:top+2*h//3, left:left+2*w//3]
        
        # 4. Resizing
        # PILのresizeだとアンチエイリアスが使えない？
        #img = img.resize((insize, insize), Image.LANCZOS)
        #print(type(img)) # PIL.Image.Image

        # 5. Converting format of chainer (np.ndarray, float32, CHW)
        array = np.asarray(img, dtype=np.float32)
        if array.ndim == 2:
            img = image[:, :, np.newaxis] # image is greyscale
        img = array.transpose(2, 0, 1)
        
        # 6. Nomilizing in [-1, 1]
        img = (img - 128.0) / 128.0 
    
        # 7. Random horizontal flipping
        if np.random.randint(2):
            img = img[:,:,::-1]

        return img

    
# メモリの有効活用のため、事前にリサイズを行っておく関数
def resize_data(paths, root='.', cashe_dir='/tmp/cashe'):

    if isinstance(paths, six.string_types):
        with open(paths) as paths_file:
            paths = [path.strip() for path in paths_file]

    if not os.path.exists(cashe_dir):
        os.mkdir(cashe_dir)

    output_paths = []
    for path in paths:
        img_file = os.path.join(root, path)
        #print(img_file)
        
        img_split = img_file.split('/')
        output_dir = os.path.join(cashe_dir, img_split[-2])
        output_file = os.path.join(output_dir, img_split[-1])
        #print(output_file)
        output_paths.append(output_file)

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        if not os.path.exists(output_file):
            img = Image.open(path, 'r')
            # Crop center
            if img.size[0] > img.size[1]:
                sub = (img.size[0] - img.size[1]) // 2
                img = img.crop((sub, 0, img.size[0]-1-sub, img.size[1]-1))
                print(img.size)
            elif img.size[0] < img.size[1]:
                sub = (img.size[1] - img.size[0]) // 2
                img = img.crop((0, sub, img.size[0]-1, img.size[1]-1-sub))
                print(img.size)
            # Resize
            img = img.resize((insize, insize), Image.ANTIALIAS)
            img.save(output_file)

    return output_paths
                    
        
    
def main():
    parser = argparse.ArgumentParser(description='Chainer example: DCGAN')
    parser.add_argument('--batchsize', '-b', type=int, default=50, # default=50
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=500, # defalt=500
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--dataset', '-i', default='train_all.txt', # defalt=''
                        help='Directory of image files.  Default is cifar-10.')
    parser.add_argument('--out', '-o', default='result_anime', # defalt='result'
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
        resized_paths = resize_data(paths=args.dataset)
        train = PreprocessedDataset(paths=resized_paths, root='/')
        print('{} contains {} image files'
              .format(args.dataset, train.__len__()))

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)

    Set up a trainer
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

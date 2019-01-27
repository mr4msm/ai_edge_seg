# -*- coding: utf-8 -*-

import multiprocessing as mp
import os
import sys
from argparse import ArgumentParser

from chainer import optimizers as optims
from chainer import serializers as S
from chainer.optimizer import GradientClipping, WeightDecay

from batch_generator import BatchGeneratorForAiEdgeSegmentation
from loss import Loss
from misc import argv2string
from model import Model
from train import train


def main():
    parser = ArgumentParser()

    parser.add_argument(
        'train_data', help='train data'
    )
    parser.add_argument(
        'train_labels', help='train labels'
    )
    parser.add_argument(
        '--val-data', default=None, help='val data'
    )
    parser.add_argument(
        '--val-labels', default=None, help='val labels'
    )
    parser.add_argument(
        '-b', '--batch-size', type=int, default=5,
        help='mini-batch size (default=5)'
    )
    parser.add_argument(
        '--beta2', type=float, default=0.999,
        help='beta2 of Adam (default=0.999)'
    )
    parser.add_argument(
        '-g', '--gpu-id', type=int, default=-1,
        help='GPU ID (default=-1, indicates CPU)'
    )
    parser.add_argument(
        '--ignore-labels', type=int, default=[], nargs='+',
        help='labels to ignore (default=[])'
    )
    parser.add_argument(
        '-l', '--learning-rate', type=float, default=0.1,
        help='learning rate (default=0.1)'
    )
    parser.add_argument(
        '--max-iter', type=int, default=160000,
        help='train model up to max-iter (default=160000)'
    )
    parser.add_argument(
        '--mean-interval', type=int, default=1000,
        help='calculate mean of train/loss (and validation loss) ' +
        'every mean-interval iters (default=1000)'
    )
    parser.add_argument(
        '--model', default=None,
        help='resume to train the model'
    )
    parser.add_argument(
        '--momentum', type=float, default=0.9,
        help='momentum rate (default=0.9)'
    )
    parser.add_argument(
        '--n-classes', type=int, default=5,
        help='number of classes (default=5)'
    )
    parser.add_argument(
        '--noise', default='no',
        help='noise injection method. \'no\', \'patch\', ' +
        'and \'permutation\' are available (default=\'no\')'
    )
    parser.add_argument(
        '--optim', default='nesterov',
        help='optimization method. \'sgd\', \'nesterov\', ' +
        'and \'adam\' are available (default=\'nesterov\')'
    )
    parser.add_argument(
        '-o', '--outdir', default='./',
        help='trained models and optimizer states are stored in outdir ' +
        '(default=\'./\')'
    )
    parser.add_argument(
        '--queue-maxsize', type=int, default=10,
        help='maxsize of queues for training and validation (default=10)'
    )
    parser.add_argument(
        '--save-interval', type=int, default=10000,
        help='save model & optimizer every save-interval iters (default=10000)'
    )
    parser.add_argument(
        '--state', default=None,
        help='optimizer state. resume to train the model with the optimizer'
    )
    parser.add_argument(
        '-w', '--weight-decay', type=float, default=1e-4,
        help='weight decay factor (default=1e-4)'
    )

    args = parser.parse_args()

    print(argv2string(sys.argv) + '\n')
    for arg in dir(args):
        if arg[:1] == '_':
            continue
        print('{} = {}'.format(arg, getattr(args, arg)))
    print()

    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)
        print('mkdir ' + args.outdir + '\n')

    model = Model(in_ch=3, out_ch=args.n_classes)
    if args.model is not None:
        S.load_npz(args.model, model)
    loss_func = Loss(model)

    if args.optim.lower() in 'sgd':
        if args.momentum > 0:
            optim = optims.CorrectedMomentumSGD(
                lr=args.learning_rate, momentum=args.momentum)
        else:
            optim = optims.SGD(lr=args.learning_rate)
    elif args.optim.lower() in 'nesterovag':
        optim = optims.NesterovAG(
            lr=args.learning_rate, momentum=args.momentum)
    elif args.optim.lower() in 'adam':
        optim = optims.Adam(
            alpha=args.learning_rate, beta1=args.momentum,
            beta2=args.beta2, weight_decay_rate=args.weight_decay,
            amsgrad=True)
    else:
        raise ValueError('Please specify an available optimizer name.\n' +
                         'SGD, NesterovAG, and Adam are available.')

    print('{}\n'.format(type(optim)))
    optim.setup(model)

    if args.state is not None:
        S.load_npz(args.state, optim)

    if (args.weight_decay > 0) and not isinstance(optim, optims.Adam):
        optim.add_hook(WeightDecay(args.weight_decay))

    optim.add_hook(GradientClipping(1))

    lr_decay_iter_dict = {int(5 * args.max_iter / 8): 0.1,
                          int(7 * args.max_iter / 8): 0.1,
                          }

    with open(args.train_data, 'r') as f:
        train_data_path_list = [line.strip() for line in f.readlines()]
    with open(args.train_labels, 'r') as f:
        train_labels_path_list = [line.strip() for line in f.readlines()]

    assert len(train_data_path_list) == len(train_labels_path_list)

    if (args.val_data is not None) or (args.val_labels is not None):
        if (args.val_data is not None) and (args.val_labels is not None):
            with open(args.val_data, 'r') as f:
                val_data_path_list = [line.strip() for line in f.readlines()]
            with open(args.val_labels, 'r') as f:
                val_labels_path_list = [line.strip() for line in f.readlines()]
            assert len(val_data_path_list) == len(val_labels_path_list)
        else:
            raise ValueError('Either val_data or val_labels is not specified.')

    train_queue = mp.Queue(maxsize=args.queue_maxsize)
    train_generator = BatchGeneratorForAiEdgeSegmentation(
        args.batch_size, train_data_path_list, train_labels_path_list,
        train_queue, train=True, noise_injection=args.noise,
        out_height=512, out_width=512,
        max_height=1216, max_width=1216,
        min_height=832, min_width=832)
    train_generator.start()

    if args.val_data is None:
        val_queue = None
    else:
        val_queue = mp.Queue(maxsize=args.queue_maxsize)
        try:
            val_generator = BatchGeneratorForAiEdgeSegmentation(
                1, val_data_path_list, val_labels_path_list, val_queue,
                train=False, out_height=608, out_width=968)
            val_generator.start()
        except Exception:
            train_generator.terminate()
            train_queue.close()
            val_queue.close()
            raise

    try:
        train(loss_func, optim, train_queue, args.max_iter, args.mean_interval,
              args.save_interval, val_queue, lr_decay_iter_dict, args.gpu_id,
              args.ignore_labels, args.outdir)
    except BaseException:
        train_generator.terminate()
        train_queue.close()
        if val_queue is not None:
            val_generator.terminate()
            val_queue.close()
        raise

    train_generator.terminate()
    train_queue.close()
    if val_queue is not None:
        val_generator.terminate()
        val_queue.close()


if __name__ == '__main__':
    main()

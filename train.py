# -*- coding: utf-8 -*-

import os
from datetime import datetime as dt

import chainer
from chainer import optimizers as optims
from chainer import serializers as S
from chainer import using_config
from chainer.backends import cuda


def train(loss_func, optim, train_queue,
          max_iter, mean_interval, save_interval,
          val_queue=None, lr_decay_iter_dict={},
          gpu_id=-1, ignore_labels=[], outdir='./'):
    chainer.global_config.train = True
    chainer.global_config.enable_backprop = True

    if gpu_id >= 0:
        loss_func.to_gpu(device=gpu_id)

    for key, value in lr_decay_iter_dict.items():
        if optim.t >= key:
            if isinstance(optim, optims.Adam):
                optim.eta *= value
            else:
                optim.lr *= value

    sum_loss = 0.
    while optim.t < max_iter:
        x_batch, t_batch, epoch_done = train_queue.get()

        for ignore_label in ignore_labels:
            t_batch[t_batch == ignore_label] = -1

        if gpu_id >= 0:
            x_batch = cuda.to_gpu(x_batch, device=gpu_id)
            t_batch = cuda.to_gpu(t_batch, device=gpu_id)

        loss = loss_func(x_batch, t_batch)
        loss_func.cleargrads()
        loss.backward()

        optim.update()
        sum_loss += float(loss.array)

        if epoch_done:
            optim.new_epoch()

        print(dt.now())
        print('epoch: {0:04d}, iter: {1:07d}, lr: {2:e}'.format(
            optim.epoch, optim.t, optim.lr))
        print('train/loss: {}'.format(float(loss.array)))

        if optim.t in lr_decay_iter_dict:
            if isinstance(optim, optims.Adam):
                optim.eta *= lr_decay_iter_dict[optim.t]
            else:
                optim.lr *= lr_decay_iter_dict[optim.t]

        if optim.t % mean_interval == 0:
            print('mean train/loss: {}'.format(sum_loss / mean_interval))
            sum_loss = 0.

            if val_queue is not None:
                val_loss = 0.
                val_valid_size = 0

                with using_config('train', False), \
                        using_config('enable_backprop', False):
                    while True:
                        x_batch, t_batch, epoch_done = val_queue.get()

                        for ignore_label in ignore_labels:
                            t_batch[t_batch == ignore_label] = -1

                        if len(ignore_labels) > 0:
                            valid_size = (t_batch != -1).sum()
                        else:
                            valid_size = t_batch.size

                        val_valid_size += valid_size

                        if gpu_id >= 0:
                            x_batch = cuda.to_gpu(x_batch, device=gpu_id)
                            t_batch = cuda.to_gpu(t_batch, device=gpu_id)

                        loss = cuda.to_cpu(loss_func(x_batch, t_batch).array)
                        loss *= valid_size

                        val_loss += loss

                        if epoch_done:
                            break

                print('val/loss: {}'.format(val_loss / val_valid_size))

        if optim.t % save_interval == 0:
            save_dst_path = os.path.join(
                outdir, 'model_iter_{0:07d}.npz'.format(optim.t))
            S.save_npz(save_dst_path, optim.target)
            print('save ' + save_dst_path)

            save_dst_path = os.path.join(
                outdir, 'optim_iter_{0:07d}.npz'.format(optim.t))
            S.save_npz(save_dst_path, optim)
            print('save ' + save_dst_path)

        print()

    if optim.t % mean_interval > 0:
        print('mean train/loss: {}'.format(
            sum_loss / (optim.t % mean_interval)))

    if optim.t % save_interval > 0:
        save_dst_path = os.path.join(
            outdir, 'model_iter_{0:07d}.npz'.format(optim.t))
        S.save_npz(save_dst_path, optim.target)
        print('save ' + save_dst_path)

        save_dst_path = os.path.join(
            outdir, 'optim_iter_{0:07d}.npz'.format(optim.t))
        S.save_npz(save_dst_path, optim)
        print('save ' + save_dst_path)

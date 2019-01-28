# -*- coding: utf-8 -*-

import multiprocessing as mp

import cv2
import numpy as np

from predict import (mean_std_in_limited_range,
                     normalize_with_limited_range_stat)


def make_random_patch(shape,
                      min_size_ratio=0.02, max_size_ratio=0.4,
                      max_aspect_ratio_h=3., max_aspect_ratio_w=3.):
    height, width = shape[:2]
    area = height * width
    patch_area = np.random.uniform(
        low=min_size_ratio, high=max_size_ratio) * area

    aspect_ratio = np.random.uniform(
        low=1., high=max_aspect_ratio_w) / np.random.uniform(
            low=1., high=max_aspect_ratio_h)
    patch_height = np.sqrt(patch_area / aspect_ratio)
    patch_width = min(int(aspect_ratio * patch_height), width)
    patch_height = min(int(patch_height), height)

    top = np.random.randint(0, height - patch_height + 1)
    left = np.random.randint(0, width - patch_width + 1)

    shape = (patch_height, patch_width) + shape[2:]
    return top, left, np.random.randint(0, 256, size=shape, dtype='uint8')


def permute_pixels_in_channel_wise(
        image_in_chw, ratio=0.5,
        min_size_ratio=0.02, max_size_ratio=0.4,
        max_aspect_ratio_h=3., max_aspect_ratio_w=3.):
    if np.random.uniform() >= ratio:
        return image_in_chw

    image_in_chw = image_in_chw.copy()
    height, width = image_in_chw.shape[1:]
    area = height * width
    patch_area = np.random.uniform(
        low=min_size_ratio, high=max_size_ratio) * area

    aspect_ratio = np.random.uniform(
        low=1., high=max_aspect_ratio_w) / np.random.uniform(
            low=1., high=max_aspect_ratio_h)
    patch_height = np.sqrt(patch_area / aspect_ratio)
    patch_width = min(int(aspect_ratio * patch_height), width)
    patch_height = min(int(patch_height), height)

    top = np.random.randint(0, height - patch_height + 1)
    left = np.random.randint(0, width - patch_width + 1)
    bottom = top + patch_height
    right = left + patch_width

    for ch in range(len(image_in_chw)):
        image_in_chw[ch, top:bottom, left:right] = np.random.permutation(
            image_in_chw[ch, top:bottom, left:right].ravel()).reshape(
                (patch_height, patch_width))

    return image_in_chw


def convert_label_to_one_hot(t, n_classes=20, dtype='uint8'):
    one_hot = np.zeros(t.shape + (n_classes,), dtype=dtype)
    for label in range(n_classes):
        one_hot[t == label, label] = 1
    return one_hot


class BatchGenerator(mp.Process):
    def __init__(self, batch_size, data_path_list,
                 labels_path_list, queue,
                 train=True, noise_injection='no',
                 out_height=None, out_width=None,
                 max_height=None, max_width=None,
                 min_height=None, min_width=None):
        super(BatchGenerator, self).__init__()
        self.batch_size = batch_size
        self.data_path_list = np.asarray(data_path_list)
        self.labels_path_list = np.asarray(labels_path_list)
        self.queue = queue
        self.train = train

        self.use_random_erasing = False
        self.use_channel_permutation = False

        if noise_injection in 'patch':
            self.use_random_erasing = True
            print('Use random erasing as noise injection in {}.\n'.format(
                'train' if train else 'val/test'
            ))
        elif noise_injection in 'permutation':
            self.use_channel_permutation = True
            print(
                'Use channel-wise permutation as noise injection ' +
                'in {}.\n'.format('train' if train else 'val/test'))
        else:
            print('No noise injection in {}.\n'.format(
                'train' if train else 'val/test'))

        example_image = cv2.imread(data_path_list[0], cv2.IMREAD_UNCHANGED)

        if out_height is None:
            self.out_height = example_image.shape[0]
        else:
            self.out_height = out_height

        if out_width is None:
            self.out_width = example_image.shape[1]
        else:
            self.out_width = out_width

        if max_height is None:
            self.max_height = example_image.shape[0]
        else:
            self.max_height = min(max_height, example_image.shape[0])

        if max_width is None:
            self.max_width = example_image.shape[1]
        else:
            self.max_width = min(max_width, example_image.shape[1])

        if min_height is None:
            self.min_height = example_image.shape[0]
        else:
            self.min_height = max(min_height, 1)

        if min_width is None:
            self.min_width = example_image.shape[1]
        else:
            self.min_width = max(min_width, 1)

    def run(self):
        if self.train:
            self.__run_train()
        else:
            self.__run_test()

    def __run_train(self):
        st_idx = 0
        indices = list(np.random.permutation(len(self.data_path_list)))
        while(True):
            current_indices = []
            epoch_done = False
            while (len(current_indices) < self.batch_size):
                ed_idx = min(st_idx + self.batch_size -
                             len(current_indices), len(indices))
                current_indices += indices[st_idx: ed_idx]
                st_idx = ed_idx % len(indices)
                if st_idx == 0:
                    epoch_done = True
                    indices = list(np.random.permutation(
                        len(self.data_path_list)))

            x = [cv2.imread(data_path, cv2.IMREAD_COLOR) for
                 data_path in self.data_path_list[current_indices]]

            mean_std_list = np.asarray(
                [mean_std_in_limited_range(x_tmp) for x_tmp in x])
            mean_list = mean_std_list[:, 0]
            std_list = mean_std_list[:, 1]

            height_list = np.random.randint(
                self.min_height, self.max_height + 1,
                size=self.batch_size)
            width_list = np.random.randint(
                self.min_width, self.max_width + 1, size=self.batch_size)

            top_list = [np.random.randint(0, image.shape[0] - height + 1)
                        for image, height in zip(x, height_list)]
            left_list = [np.random.randint(0, image.shape[1] - width + 1)
                         for image, width in zip(x, width_list)]

            horizontal_flip = np.random.randint(0, 2, size=self.batch_size)
            horizontal_flip[horizontal_flip == 0] = -1

            x = [cv2.resize(
                ((x_tmp[top: top + height, left: left + width] - mean) /
                 (std + 1e-8)), (self.out_width, self.out_height),
                interpolation=cv2.INTER_LINEAR)[:, ::flip]
                for x_tmp, left, top, width, height, mean, std, flip in zip(
                x, left_list, top_list, width_list, height_list,
                mean_list, std_list, horizontal_flip)
            ]

            if self.use_random_erasing:
                use_list = np.random.randint(
                    0, 2, size=self.batch_size).astype(bool)

                for idx, use in enumerate(use_list):
                    if not use:
                        continue

                    top, left, patch = make_random_patch(x[idx].shape)
                    patch = (patch - mean_list[idx]) / (std_list[idx] + 1e-8)
                    bottom = top + patch.shape[0]
                    right = left + patch.shape[1]
                    x[idx][top: bottom, left: right] = patch

            x = np.asarray(x, dtype='float32').transpose(0, 3, 1, 2)

            if self.use_channel_permutation:
                x = np.asarray([permute_pixels_in_channel_wise(x_tmp)
                                for x_tmp in x])

            t = [cv2.resize(cv2.imread(
                labels_path, cv2.IMREAD_UNCHANGED
            )[top: top + height, left: left + width],
                (self.out_width, self.out_height),
                interpolation=cv2.INTER_NEAREST)[:, ::flip]
                for labels_path, left, top, width, height, flip in zip(
                self.labels_path_list[current_indices],
                left_list, top_list, width_list, height_list,
                horizontal_flip)
            ]

            t = np.asarray(t, dtype='int32')

            self.queue.put((x, t, epoch_done))

    def __run_test(self):
        st_idx = 0
        indices = list(np.arange(len(self.data_path_list)))
        while(True):
            ed_idx = min(st_idx + self.batch_size, len(indices))
            current_indices = indices[st_idx: ed_idx]
            st_idx = ed_idx % len(indices)

            x = [normalize_with_limited_range_stat(
                cv2.imread(data_path, cv2.IMREAD_COLOR)) for
                data_path in self.data_path_list[current_indices]]

            x = [cv2.resize(x_tmp, (self.out_width, self.out_height),
                            interpolation=cv2.INTER_LINEAR) for x_tmp in x]

            t = [cv2.resize(
                cv2.imread(labels_path, cv2.IMREAD_UNCHANGED),
                (self.out_width, self.out_height),
                interpolation=cv2.INTER_NEAREST)
                for labels_path in self.labels_path_list[current_indices]]

            x = np.asarray(x, dtype='float32').transpose(0, 3, 1, 2)
            t = np.asarray(t, dtype='int32')

            self.queue.put((x, t, (st_idx == 0)))

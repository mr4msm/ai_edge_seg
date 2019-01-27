# -*- coding: utf-8 -*-

import os
import sys
from argparse import ArgumentParser

import chainer
import cv2
import numpy as np
from chainer import serializers as S
from chainer.backends import cuda

from misc import argv2string
from model import Model

class_rgb_list = [
    (0, 0, 255), (193, 214, 0), (180, 0, 129), (255, 121, 166), (255, 0, 0),
    (65, 166, 1), (208, 149, 1), (255, 255, 0), (255, 134, 0), (0, 152, 225),
    (0, 203, 151), (85, 255, 50), (92, 136, 125), (69, 47, 142), (136, 45, 66),
    (0, 255, 255), (215, 0, 255), (180, 131, 135), (81, 99, 0), (86, 62, 67),
]


def mean_std_in_limited_range(image_bgr, lower=0, upper=192):
    if image_bgr.ndim == 2:
        valid = ((image_bgr >= lower) & (image_bgr < upper))
        return image_bgr[valid].mean(), image_bgr[valid].std()

    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    valid = ((image_gray >= lower) & (image_gray < upper))
    return image_gray[valid].mean(), image_gray[valid].std()


def normalize_with_limited_range_stat(x, lower=0, upper=192):
    mean, std = mean_std_in_limited_range(x, lower=lower, upper=upper)
    return (x - mean) / (std + 1e-8)


class Predictor(object):
    def __init__(self, model):
        self.model = model

        for param in model.params(include_uninit=False):
            if isinstance(param.array, np.ndarray):
                self.gpu_id = -1
            else:
                self.gpu_id = param.array.device.id
            break

    def predict(self, image_bgr):
        chainer.global_config.train = False
        chainer.global_config.enable_backprop = False

        height, width = image_bgr.shape[:2]
        normed_image = normalize_with_limited_range_stat(image_bgr)
        resized_image = cv2.resize(normed_image, (width // 2, height // 2),
                                   interpolation=cv2.INTER_LINEAR)
        x = resized_image.transpose(2, 0, 1).astype('float32')
        x = x[np.newaxis]

        if self.gpu_id >= 0:
            x = cuda.to_gpu(x, device=self.gpu_id)
        y = cuda.to_cpu(self.model(x).array)[0]

        exp = np.exp(y.transpose(1, 2, 0))
        softmax = exp / exp.sum(axis=2, keepdims=True)
        prob_map = cv2.resize(softmax, (width, height),
                              interpolation=cv2.INTER_NEAREST)
        prob_map = cv2.blur(prob_map, (3, 3), borderType=cv2.BORDER_CONSTANT)

        return prob_map.argmax(axis=2)


def main():
    parser = ArgumentParser()

    parser.add_argument(
        'input_image_path'
    )
    parser.add_argument(
        '-g', '--gpu-id', type=int, default=-1,
        help='GPU ID (default=-1, indicates CPU)'
    )
    parser.add_argument(
        '-m', '--model', default=None,
        help='trained model (param)'
    )
    parser.add_argument(
        '-n', '--n-classes', type=int, default=5,
        help='number of classes (default=5)'
    )
    parser.add_argument(
        '-o', '--output', default=None
    )

    args = parser.parse_args()

    print(argv2string(sys.argv) + '\n')
    for arg in dir(args):
        if arg[:1] == '_':
            continue
        print('{} = {}'.format(arg, getattr(args, arg)))
    print()

    if args.output is None:
        wo_ext = os.path.splitext(args.input_image_path)[0]
        out_path = wo_ext + '_segmented' + '.png'
    else:
        out_path = args.output

    out_dir, out_bname = os.path.split(out_path)

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
        print('mkdir ' + args.outdir + '\n')

    model = Model(in_ch=3, out_ch=args.n_classes)
    if args.model is not None:
        S.load_npz(args.model, model)

    if args.gpu_id >= 0:
        model.to_gpu(device=args.gpu_id)

    predictor = Predictor(model)

    image_bgr = cv2.imread(args.input_image_path, cv2.IMREAD_COLOR)
    label_map = predictor.predict(image_bgr)

    if args.n_classes == 5:
        table_to_convert = (0, 4, 7, 13, 18)
        label_map_tmp = label_map
        label_map = np.empty_like(label_map)
        for idx, to_label in enumerate(table_to_convert):
            label_map[label_map_tmp == idx] = to_label

    label_rgb = np.zeros_like(image_bgr)
    for idx, class_rgb in enumerate(class_rgb_list):
        label_rgb[label_map == idx] = class_rgb

    cv2.imwrite(out_path, label_rgb[:, :, ::-1])


if __name__ == '__main__':
    main()

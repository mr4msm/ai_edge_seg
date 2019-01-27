# -*- coding: utf-8 -*-

import numpy as np
from chainer import Chain
from chainer import functions as F
from chainer.backends import cuda


class Loss(Chain):
    def __init__(self, model):
        super(Loss, self).__init__()

        self.class_weights = self.xp.asarray(
            [0.625, 1.25, 2.5, 0.3125, 0.3125])

        with self.init_scope():
            self.model = model

    def forward(self, x, t, ignore_label=-1):
        y = self.model(x)
        y = y.transpose((0, 2, 3, 1))
        y = y.reshape((-1, y.shape[-1]))

        y_exp = self.xp.exp(y.array)
        y_softmax = y_exp / y_exp.sum(axis=1, keepdims=True)

        y = F.log_softmax(y)

        t = t.ravel()
        t_valid = (t != ignore_label)
        t *= t_valid

        focal_weights = self.class_weights[t] * \
            (1 - y_softmax[np.arange(t.size), t])

        loss = y[np.arange(t.size), t] * focal_weights
        loss *= t_valid

        return -F.sum(loss) / t_valid.sum()

    def to_cpu(self):
        super(Loss, self).to_cpu()
        self.class_weights = cuda.to_cpu(self.class_weights)

    def to_gpu(self, device=None):
        super(Loss, self).to_gpu(device=device)
        self.class_weights = cuda.to_gpu(self.class_weights, device=device)

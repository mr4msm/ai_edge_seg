# -*- coding: utf-8 -*-

from chainer import Chain, Sequential
from chainer import functions as F
from chainer import initializers as inits
from chainer import links as L

from automatic_batch_renormalization import AutomaticBatchRenormalization


class ObjectContextPooling(Chain):
    def __init__(self, in_ch, key_ch, out_ch, w_init=inits.HeNormal()):
        super(ObjectContextPooling, self).__init__()

        with self.init_scope():
            self.f_key = Sequential(
                L.Convolution2D(
                    in_ch, key_ch, ksize=1, pad=0,
                    nobias=True, initialW=w_init),
                AutomaticBatchRenormalization(key_ch, decay=0.99)
            )
            self.f_value = L.Convolution2D(
                in_ch, out_ch, ksize=1, pad=0, nobias=False, initialW=w_init)

    def forward(self, x):
        key = self.f_key(x)
        b, c, h, w = key.shape
        key = key.reshape((b, c, h * w))
        query = key.transpose((0, 2, 1))

        sim_map = F.matmul(query, key)
        sim_map *= (c ** -0.5)
        sim_map = F.softmax(sim_map, axis=1)

        value = self.f_value(x)
        b, c, h, w = value.shape
        value = value.reshape((b, c, h * w))

        context = F.matmul(value, sim_map)
        context = context.reshape((b, c, h, w))

        return context


class OCBlock(Chain):
    def __init__(self, in_ch, out_ch, w_init=inits.HeNormal()):
        super(OCBlock, self).__init__()

        with self.init_scope():
            self.conv_pre = Sequential(
                L.Convolution2D(
                    in_ch, out_ch, ksize=3, pad=1,
                    nobias=True, initialW=w_init),
                AutomaticBatchRenormalization(out_ch, decay=0.99)
            )
            self.context = ObjectContextPooling(
                out_ch, out_ch // 2, out_ch, w_init=w_init)
            self.conv_post = Sequential(
                L.Convolution2D(
                    2 * out_ch, out_ch, ksize=1, pad=0,
                    nobias=True, initialW=w_init),
                AutomaticBatchRenormalization(out_ch, decay=0.99)
            )

    def forward(self, x):
        y = self.conv_pre(x)
        y = F.concat((y, self.context(y)), axis=1)
        y = self.conv_post(y)

        return y

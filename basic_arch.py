# -*- coding: utf-8 -*-

from chainer import Chain, Sequential
from chainer import functions as F
from chainer import initializers as inits
from chainer import links as L

from automatic_batch_renormalization import AutomaticBatchRenormalization


class Unit(Sequential):
    def __init__(self, in_ch, out_ch_list,
                 ksize_list=None,
                 use_bn_list=None,
                 activation_list=None,
                 w_init=inits.HeNormal()):
        if ksize_list is None:
            ksize_list = [3] * len(out_ch_list)

        if use_bn_list is None:
            use_bn_list = [True] * len(out_ch_list)

        if activation_list is None:
            if len(out_ch_list) == 1:
                activation_list = [F.relu]
            else:
                activation_list = [None] + [F.relu] * (len(out_ch_list) - 1)

        assert len(out_ch_list) == len(ksize_list), \
            (len(out_ch_list), len(ksize_list))

        assert len(use_bn_list) == len(activation_list), \
            (len(use_bn_list), len(activation_list))

        assert len(out_ch_list) == len(use_bn_list), \
            (len(out_ch_list), len(use_bn_list))

        super(Unit, self).__init__()

        in_ch_list = [in_ch] + out_ch_list[:-1]

        for in_ch, out_ch, ksize, use_bn, activation in zip(
                in_ch_list, out_ch_list, ksize_list,
                use_bn_list, activation_list):
            if use_bn:
                self.append(AutomaticBatchRenormalization(in_ch, decay=0.99))
            if activation is not None:
                self.append(activation)
            self.append(L.Convolution2D(in_ch, out_ch, ksize=ksize,
                                        pad=(ksize - 1) // 2,
                                        nobias=True, initialW=w_init))
        self.append(
            AutomaticBatchRenormalization(out_ch_list[-1], decay=0.99))


class Residual(Chain):
    def __init__(self, network):
        super(Residual, self).__init__()

        with self.init_scope():
            self.network = network

    def forward(self, x):
        y = self.network(x)
        bsize, in_ch, height, width = x.shape
        out_ch = y.shape[1]

        if in_ch < out_ch:
            pad = self.xp.zeros((bsize, out_ch - in_ch, height, width),
                                dtype=x.dtype)
            skip = F.concat((x, pad), axis=1)
        elif in_ch > out_ch:
            skip = x[:, :out_ch]
        else:
            skip = x

        return skip + y


class Block(Sequential):
    def __init__(self, in_ch, out_ch_list,
                 n_convs_list=None,
                 ksize_list=None,
                 use_bn_list=None,
                 activation_list=None,
                 w_init=inits.HeNormal()):
        if n_convs_list is None:
            n_convs_list = [2] * len(out_ch_list)

        if ksize_list is None:
            ksize_list = [3] * len(out_ch_list)

        if use_bn_list is None:
            use_bn_list = [True] * len(out_ch_list)

        if activation_list is None:
            activation_list = [F.relu] * len(out_ch_list)

        assert len(out_ch_list) == len(n_convs_list), \
            (len(out_ch_list), len(n_convs_list))

        assert len(n_convs_list) == len(ksize_list), \
            (len(n_convs_list), len(ksize_list))

        assert len(ksize_list) == len(use_bn_list), \
            (len(ksize_list), len(use_bn_list))

        assert len(use_bn_list) == len(activation_list), \
            (len(use_bn_list), len(activation_list))

        super(Block, self).__init__()

        in_ch_list = [in_ch] + out_ch_list[:-1]

        for in_ch, out_ch, n_convs, ksize, use_bn, activation in zip(
                in_ch_list, out_ch_list, n_convs_list,
                ksize_list, use_bn_list, activation_list):
            out_ch_list_for_unit = [
                in_ch + (out_ch - in_ch) * (idx + 1) // n_convs
                for idx in range(n_convs)]

            self.append(Residual(Unit(
                in_ch=in_ch, out_ch_list=out_ch_list_for_unit,
                ksize_list=[ksize] * n_convs, use_bn_list=[use_bn] * n_convs,
                activation_list=[None] + [activation] * (n_convs - 1),
                w_init=w_init)))

# -*- coding: utf-8 -*-

from chainer import Chain, Sequential
from chainer import functions as F
from chainer import initializers as inits
from chainer import links as L

from automatic_batch_renormalization import AutomaticBatchRenormalization
from basic_arch import Block, Residual, Unit
from object_context import OCBlock


class Model(Chain):
    def __init__(self, in_ch, out_ch):
        super(Model, self).__init__()

        w_init = inits.HeNormal()

        with self.init_scope():
            self.conv0 = L.Convolution2D(
                in_ch, 12, ksize=4, stride=2, pad=1,
                nobias=True, initialW=w_init)
            self.bn0 = AutomaticBatchRenormalization(12, decay=0.99)

            self.conv1 = Sequential(Residual(Unit(
                in_ch=12, out_ch_list=[15, 18],
                use_bn_list=[False, True], w_init=w_init)))
            self.conv1.extend(Block(
                in_ch=18, out_ch_list=[24], w_init=w_init))

            self.conv2 = Block(
                in_ch=24, out_ch_list=[30, 36], w_init=w_init)
            self.conv3 = Block(
                in_ch=36, out_ch_list=[42, 48], w_init=w_init)
            self.conv4 = Block(
                in_ch=48, out_ch_list=[54, 60], w_init=w_init)
            self.conv5 = Block(
                in_ch=60, out_ch_list=[60, 60], w_init=w_init)
            self.conv6 = Block(
                in_ch=60, out_ch_list=[60, 60], w_init=w_init)
            self.conv7 = Block(
                in_ch=60, out_ch_list=[66, 72], w_init=w_init)

            self.oc7 = OCBlock(72, 12, w_init=w_init)

            self.conv6_branch = Block(
                in_ch=60, out_ch_list=[66, 72], w_init=w_init)
            self.oc6 = OCBlock(72, 12, w_init=w_init)

            self.conv5_branch = Block(
                in_ch=60, out_ch_list=[66, 72], w_init=w_init)
            self.oc5 = OCBlock(72, 12, w_init=w_init)

            self.conv4_branch = Block(
                in_ch=60, out_ch_list=[66, 72], w_init=w_init)
            self.oc4 = OCBlock(72, 12, w_init=w_init)

            self.conv3_branch = Block(
                in_ch=48, out_ch_list=[54, 60], w_init=w_init)
            self.oc3 = OCBlock(60, 10, w_init=w_init)

            self.conv2_branch = Block(
                in_ch=36, out_ch_list=[42, 48], w_init=w_init)
            self.conv2_out = L.Convolution2D(
                48, 8, ksize=3, pad=1, nobias=True, initialW=w_init)
            self.bn2_out = AutomaticBatchRenormalization(8, decay=0.99)

            self.conv1_branch = Block(
                in_ch=24, out_ch_list=[30, 36], w_init=w_init)
            self.conv1_out = L.Convolution2D(
                36, 6, ksize=3, pad=1, nobias=True, initialW=w_init)
            self.bn1_out = AutomaticBatchRenormalization(6, decay=0.99)

            self.conv_out = L.Convolution2D(
                4 * 12 + 10 + 8 + 6, out_ch,
                ksize=3, pad=1, nobias=False, initialW=w_init)

    def forward(self, x):
        orig_hw = x.shape[2:]
        y = self.bn0(self.conv0(x))
        half_hw = y.shape[2:]

        conv1 = self.conv1(y)
        y = F.average_pooling_2d(conv1, 2)

        conv2 = self.conv2(y)
        y = F.average_pooling_2d(conv2, 2)

        conv3 = self.conv3(y)
        y = F.average_pooling_2d(conv3, 2)

        conv4 = self.conv4(y)
        y = F.average_pooling_2d(conv4, 2)

        conv5 = self.conv5(y)
        y = F.average_pooling_2d(conv5, 2)

        conv6 = self.conv6(y)
        y = F.average_pooling_2d(conv6, 2)

        conv7 = F.relu(self.conv7(y))
        conv7 = F.relu(self.oc7(conv7))
        conv7 = F.unpooling_2d(conv7, ksize=64,
                               outsize=half_hw, cover_all=False)

        conv6 = F.relu(self.conv6_branch(conv6))
        conv6 = F.relu(self.oc6(conv6))
        conv6 = F.unpooling_2d(conv6, ksize=32,
                               outsize=half_hw, cover_all=False)

        conv5 = F.relu(self.conv5_branch(conv5))
        conv5 = F.relu(self.oc5(conv5))
        conv5 = F.unpooling_2d(conv5, ksize=16,
                               outsize=half_hw, cover_all=False)

        conv4 = F.relu(self.conv4_branch(conv4))
        conv4 = F.relu(self.oc4(conv4))
        conv4 = F.unpooling_2d(conv4, ksize=8,
                               outsize=half_hw, cover_all=False)

        conv3 = F.relu(self.conv3_branch(conv3))
        conv3 = F.relu(self.oc3(conv3))
        conv3 = F.unpooling_2d(conv3, ksize=4,
                               outsize=half_hw, cover_all=False)

        conv2 = F.relu(self.conv2_branch(conv2))
        conv2 = F.relu(self.bn2_out(self.conv2_out(conv2)))
        conv2 = F.unpooling_2d(conv2, ksize=2,
                               outsize=half_hw, cover_all=False)

        conv1 = F.relu(self.conv1_branch(conv1))
        conv1 = F.relu(self.bn1_out(self.conv1_out(conv1)))

        y = F.concat((conv1, conv2, conv3, conv4, conv5, conv6, conv7), axis=1)
        y = F.unpooling_2d(self.conv_out(y), ksize=2,
                           outsize=orig_hw, cover_all=False)

        return y

# -*- coding: utf-8 -*-

import chainer
from chainer import links as L


class AutomaticBatchRenormalization(L.BatchRenormalization):
    def __init__(self, size, rmax_init=1, rmax_last=3,
                 rmax_change_start_step=5000,
                 rmax_change_end_step=40000,
                 dmax_init=0, dmax_last=5,
                 dmax_change_start_step=5000,
                 dmax_change_end_step=25000,
                 decay=0.9, eps=2e-5,
                 dtype=None, use_gamma=True, use_beta=True,
                 initial_gamma=None, initial_beta=None,
                 initial_avg_mean=None, initial_avg_var=None):

        self.rmax_init = rmax_init
        self.rmax_last = rmax_last
        self.rmax_change_start_step = rmax_change_start_step
        self.rmax_change_end_step = rmax_change_end_step

        self.rmax_width = self.rmax_last - self.rmax_init
        self.rmax_step_size = self.rmax_change_end_step - \
            self.rmax_change_start_step

        self.dmax_init = dmax_init
        self.dmax_last = dmax_last
        self.dmax_change_start_step = dmax_change_start_step
        self.dmax_change_end_step = dmax_change_end_step

        self.dmax_width = self.dmax_last - self.dmax_init
        self.dmax_step_size = self.dmax_change_end_step - \
            self.dmax_change_start_step

        super(AutomaticBatchRenormalization, self).__init__(
            size, self.rmax_init, self.dmax_init,
            decay, eps, dtype, use_gamma, use_beta,
            initial_gamma, initial_beta, initial_avg_mean, initial_avg_var)

        self.t = 0
        self.register_persistent('t')

    def forward(self, x, finetune=False):
        if chainer.config.train:
            rmax_curr_step = max(0, self.t - self.rmax_change_start_step)
            self.rmax = self.rmax_init + \
                float(self.rmax_width) * rmax_curr_step / self.rmax_step_size
            self.rmax = min(self.rmax, self.rmax_last)

            dmax_curr_step = max(0, self.t - self.dmax_change_start_step)
            self.dmax = self.dmax_init + \
                float(self.dmax_width) * dmax_curr_step / self.dmax_step_size
            self.dmax = min(self.dmax, self.dmax_last)

            self.t += 1

        return super(AutomaticBatchRenormalization, self).forward(
            x, finetune=finetune)

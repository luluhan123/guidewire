#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-04-16 14:26:03
# @Author  : han lulu (han.fire@foxmail.com)
# @Link    : https://github.com/luluhan123/
# @Version : $Id$

import os
import torch


class data_prefetcher(object):
    """docstring for data_prefetcher"""

    def __init__(self, loader):
        super(data_prefetcher, self).__init__()
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target, self.next_distance = next(self.loader)
            # self.next_input, self.next_target = next(self.loader)
            # sampler = next(self.loader)
            # self.next_input, self.next_target = sampler['image'], sampler['label']
        except StopIteration:
            self.next_input = None
            self.next_target = None
            self.next_distance = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_distance = self.next_distance.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        distance = self.next_distance
        self.preload()
        return input, target, distance
        # return input, target

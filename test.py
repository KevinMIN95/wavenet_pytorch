import numpy as np
from numpy.matlib import repmat
import torch
import torch.nn.functional as F
import threading

from torch import nn
from wavenet import WaveNet
import argparse
import logging
import os
import sys
import time

input = torch.tensor(
    [
        [1,2,3,4,5,6,7,8,9,10],
        [11,2,3,4,5,6,7,8,9,10],
        [1,2,3,4,5,6,7,3,4,5]
    ]
)

# h = torch.rand(3,28,10)
# print(input.size())
# net = WaveNet(n_quantize=10, dilation_depth=3, dilation_repeat=3,)

# net(input, h)







# feats = np.array([
#     [1,2,3,4,5],
#     [6,7,8,9,0]
# ])

# out = extend_time(feats,2)

class BackgroundGenerator(threading.Thread):
    """BACKGROUND GENERATOR.

    Args:
        generator (object): Generator instance.
        max_prefetch (int): Max number of prefetch.

    References:
        https://stackoverflow.com/questions/7323664/python-generator-pre-fetch

    """

    def __init__(self, max_prefetch=1):
        threading.Thread.__init__(self)
        print(max_prefetch)
        try:
            import queue
        except ImportError:
            import Queue as queue
        self.queue = queue.Queue(max_prefetch)
        # self.generator = generator
        self.daemon = True
        self.start()

    def run(self):
        """STORE ITEMS IN QUEUE."""
        # for item in self.generator:
        #     self.queue.put(item)
        self.queue.put(None)

    def next(self):
        """GET ITEM IN THE QUEUE."""
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class background(object):
    """BACKGROUND GENERATOR DECORATOR."""

    def __init__(self, max_prefetch=1):
        self.max_prefetch = max_prefetch

    def __call__(self, gen):
        def bg_generator(*args, **kwargs):
            return BackgroundGenerator(max_prefetch=self.max_prefetch)
        return bg_generator

args = {
    'asdfas':2
}
from dateutil.relativedelta import relativedelta

initial_time = time.time() - 300

print("total training time ({0} iteration) = "
                         "{1.days:02}:{1.hours:02}:{1.minutes:02}:{1.seconds:02}"
                         .format((6-3),relativedelta(
                             seconds=int(time.time() - initial_time))))


l = torch.tensor([
    [
        [1,2,3,4,5],
        [3,4,5,6,7]
    ]
]).float()

print(l.size())
s = torch.softmax(l, dim=2)
print(s)
s = s.argmax(dim=2)
print(s)
print(s.numpy())
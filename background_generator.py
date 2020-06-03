import threading
from queue import *

class BackgroundGenerator(threading.Thread):
    """BACKGROUND GENERATOR.

    Args:
        generator (object): Generator instance.
        max_prefetch (int): Max number of prefetch.

    References:
        https://stackoverflow.com/questions/7323664/python-generator-pre-fetch

    """

    def __init__(self, generator, max_prefetch=1):
        threading.Thread.__init__(self)
        self.queue = Queue(max_prefetch)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self):
        """STORE ITEMS IN QUEUE."""
        for item in self.generator:
            self.queue.put(item)
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
            return BackgroundGenerator(gen(*args, **kwargs), max_prefetch=self.max_prefetch)
        return bg_generator
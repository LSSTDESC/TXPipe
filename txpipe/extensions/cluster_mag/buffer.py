import numpy as np


class Buffer:
    """
    This is a helper for the randoms generation that buffers its output
    so it's not trying to write loads of small chunks of data out.

    It stores a buffer of things to write out later of specified size.

    It's not trying to be more generally useful at this stage - it can only
    be given sequentially contiguous chunks to write
    """
    def __init__(self, size, ref, start=0, dtype=float):
        self.ref = ref
        self.start = start
        self.buffered = 0
        self.buffer = []

    def append(self, vals):
        self.buffer.append(vals)
        self.buffered += len(vals)

        if self.buffered > self.size:
            self.write()

    def write(self):
        if not self.buffer:
            return
        out = np.concatenate(self.buffer)
        n = len(out)
        self.ref[self.start : self.start + n] = out
        self.start += n
        self.buffered = 0
        self.buffer = []



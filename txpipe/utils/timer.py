from timeit import default_timer


class Timer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.t0 = default_timer()
        self.stamps = [('START', self.t0)]

    def mark(self, label):
        t = default_timer()
        dt1 = t - self.t0
        dt2 = t - self.stamps[-1][1]
        print(f"MARK {label}  {dt1} seconds since start and {dt2} seconds since last mark")
        self.stamps.append((label,t))



class AverageMeter(object):
    """Computes and stores the average and current value.
      ref: https://github.com/chrischute/glow/blob/master/util/shell_util.py
    """
    def __init__(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

        self.val1 = 0.
        self.avg1 = 0.
        self.sum1 = 0.
        self.count1 = 0.

        self.val2 = 0.
        self.avg2 = 0.
        self.sum2 = 0.
        self.count2 = 0.

        self.val3 = 0.
        self.avg3 = 0.
        self.sum3 = 0.
        self.count3 = 0.

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def reset1(self):
        self.val1 = 0.
        self.avg1 = 0.
        self.sum1 = 0.
        self.count1 = 0.

    def reset2(self):
        self.val2 = 0.
        self.avg2 = 0.
        self.sum2 = 0.
        self.count2 = 0.

    def reset3(self):
        self.val3 = 0.
        self.avg3 = 0.
        self.sum3 = 0.
        self.count3 = 0.

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def update1(self, val, n=1):
        self.val1 = val
        self.sum1 += val * n
        self.count1 += n
        self.avg1 = self.sum1 / self.count1

    def update2(self, val, n=1):
        self.val2 = val
        self.sum2 += val * n
        self.count2 += n
        self.avg2 = self.sum2 / self.count2

    def update3(self, val, n=1):
        self.val3 = val
        self.sum3 += val * n
        self.count3 += n
        self.avg3 = self.sum3 / self.count3


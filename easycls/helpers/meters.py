from .logs import init_module_logger

class AverageMeter(object):
    """
    Computes and stores the count, sum, average and current value

    Attributes:
        name: a string name of the value.
        fmt: a format string, e.g. `:6.2f`
        val: current value
        avg: average of computed values
        sum: sum of computed values
        count: count of computed values
    """
    def __init__(self, name, fmt=':f', unit=''):
        self.name = name
        self.fmt = fmt
        self.unit = unit
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '}' + self.unit + ' ({avg' + self.fmt + '}' + self.unit + ')'
        return fmtstr.format(name=self.name, val=self.val, avg=self.avg)
        # return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    """
    Display the information of a batch

    Attributes:
        meters: meters as batch information
        prefix: a prefix string
    """
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches) # e.g. num_batches=100, batch_fmtstr='[:3d/100]'
        self.meters = meters
        self.prefix = prefix

    def batch_str(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]   # 'prefix\t[{batch:3d}/100]'
        entries += [str(meter) for meter in self.meters]
        return '\t'.join(entries)

    def display(self, batch):
        print(self.batch_str(batch))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))     # num_digits=3
        fmt = '{:' + str(num_digits) + 'd}'         # '{:3d}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'  # '[:3d/100]'

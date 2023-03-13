
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


class MetricTracker:
    def __init__(self):
        self.metrics = {}

    def reset(self):
        self.metrics = {}

    def update_metrics(self, metric_dict, batch_size=1, compute_avg=True):
        for k, v in metric_dict.items():
            if k in self.metrics and compute_avg:
                self.metrics[k].update(v, batch_size)
            else:
                self.metrics[k] = AverageMeter()
                self.metrics[k].update(v, batch_size)

    def __getitem__(self, k):
        if k in self.metrics:
            return self.metrics[k].avg
        else:
            return 0.0

    def current_metrics(self):
        return {k:v.avg for k,v in self.metrics.items()}



import numpy as np

class AverageMeter(object):
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


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, dim=-1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(1. / batch_size))
    if len(res) == 1:
        return res[0]
    return res


def shot_acc(preds, labels, training_labels, many_shot_thr=100, low_shot_thr=20):
    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    train_class_count = []
    test_class_count = []
    class_correct = []
    for l in np.unique(labels):
        train_class_count.append(len(training_labels[training_labels == l]))
        test_class_count.append(len(labels[labels == l]))
        class_correct.append((preds[labels == l] == labels[labels == l]).sum())

    many_shot = []
    median_shot = []
    low_shot = []
    for i in range(len(train_class_count)):
        if train_class_count[i] >= many_shot_thr:
            many_shot.append((class_correct[i] / test_class_count[i]))
        elif train_class_count[i] <= low_shot_thr:
            low_shot.append((class_correct[i] / test_class_count[i]))
        else:
            median_shot.append((class_correct[i] / test_class_count[i]))

    check = sum(class_correct) / sum(test_class_count)
    print('check correctness: acc: %.4f, class_correct: %d, test_class_count: %d' % (check, sum(class_correct), sum(test_class_count)))

    return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot)


def equal_to(x1, x2, eta=1e-9):
    return x1 > x2 - eta and x1 < x2 + eta


if __name__ == "__main__":
    import numpy as np
    import torch
    pred = np.array([[0.1, 0.3, 0.6],
            [0.4, 0.45, 0.15],
            [0.2, 0.6, 0.1]])
    label = np.array([0, 1, 2])
    pred = torch.from_numpy(pred)
    label = torch.from_numpy(label)
    acc = accuracy(pred, label)
    print(acc)
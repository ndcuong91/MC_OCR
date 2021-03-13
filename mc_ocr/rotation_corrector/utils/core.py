import time
from tqdm import tqdm
import numpy as np
from statistics import mean
import torch
import torch.nn as nn
import torch.nn.functional as F


def train_step(loader, net, crit, optim, dev, total_step, logging, config, debug_steps=100, epo=-1):
    net.train()
    # print((loader))
    for i, (images, labels) in enumerate(loader):
        images = images.to(dev)
        labels = labels.to(dev)
        optim.zero_grad()

        # Forward pass
        outputs = net(images)
        labels = labels.argmax(1)
        loss = crit(outputs, labels)

        # Backward and optimize
        loss.backward()
        optim.step()

        if (i + 1) % debug_steps == 0:
            logging.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                         .format(epo, config.TRAIN.END_EPOCH, i + 1, total_step, loss.item()))


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

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


def evaluation(loader, net, crit, dev, logging, thres=0.3, classNum=7):
    net.eval()
    post_pr = ClsPostProcess(['0', '180'])
    mtric = ClsMetric()
    with torch.no_grad():
        total_frame = 0.0
        total_time = 0.0
        running_loss = 0.0
        pbar = tqdm(total=len(loader), desc='eval model:')
        for idx, batch in enumerate(loader):
            if idx >= len(loader):
                break
            images = batch[0].to(dev)
            targets = batch[1].to(dev)
            start = time.time()
            preds = net(images)
            targets = targets.argmax(1)
            loss = crit(preds, targets)
            running_loss += loss.item()
            post_result = post_pr(preds.clone().cpu().numpy(), batch[1].numpy())
            total_time += time.time() - start
            mtric(post_result)
            pbar.update(1)
            total_frame += len(images)
        metirc = mtric.get_metric()

    pbar.close()
    net.train()
    metirc['fps'] = round(total_frame / total_time, 3)
    metirc['loss'] = round(running_loss / len(loader), 3)

    logging.info('val acc: {}, val loss {}, fps {}'.format(metirc['acc'], metirc['loss'], metirc['fps']))
    return metirc


class ClsMetric(object):
    def __init__(self, main_indicator='acc', **kwargs):
        self.main_indicator = main_indicator
        self.reset()

    def __call__(self, pred_label, *args, **kwargs):
        preds, labels = pred_label
        correct_num = 0
        all_num = 0
        for (pred, pred_conf), (target, _) in zip(preds, labels):
            if pred == target:
                correct_num += 1
            all_num += 1
        self.correct_num += correct_num
        self.all_num += all_num
        return {'acc': correct_num / all_num, }

    def get_metric(self):
        """
        return metrics {
                 'acc': 0
            }
        """
        acc = self.correct_num / self.all_num
        self.reset()
        return {'acc': round(acc, 2)}

    def reset(self):
        self.correct_num = 0
        self.all_num = 0


class ClsPostProcess(object):
    """ Convert between text-label and text-index """

    def __init__(self, label_list, **kwargs):
        super(ClsPostProcess, self).__init__()
        self.label_list = label_list

    def __call__(self, preds, label=None, *args, **kwargs):
        if isinstance(preds, torch.Tensor):
            preds = preds.numpy()

        pred_idxs = preds.argmax(axis=1)
        # print('1', preds)
        # print('pred_idxs',pred_idxs)
        decode_out = [(self.label_list[idx], preds[i, idx])
                      for i, idx in enumerate(pred_idxs)]
        if label is None:
            return decode_out
        label = label.argmax(1)
        label = [(self.label_list[idx], 1.0) for idx in label]
        # print(decode_out, label)
        return decode_out, label

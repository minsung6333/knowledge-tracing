import json
import os
import pprint as pp
import random
import numpy as np
from datetime import date
from pathlib import Path
from abc import ABCMeta, abstractmethod

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch import optim as optim

def save_state_dict(state_dict, path, filename):
    torch.save(state_dict, os.path.join(path, filename))

def ndcg(scores, labels, k):
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hits = labels.gather(1, cut)
    position = torch.arange(2, 2+k)
    weights = 1 / torch.log2(position.float())
    dcg = (hits.float() * weights).sum(1)
    idcg = torch.Tensor([weights[:min(int(n), k)].sum()
                         for n in labels.sum(1)])
    ndcg = dcg / idcg
    return ndcg.mean()

def absolute_recall_mrr_ndcg_for_ks(scores, labels, ks):
    metrics = {}
    labels = F.one_hot(labels, num_classes=scores.size(1))
    answer_count = labels.sum(1)

    labels_float = labels.float()
    rank = (-scores).argsort(dim=1)

    cut = rank
    for k in sorted(ks, reverse=True):
        cut = cut[:, :k]
        hits = labels_float.gather(1, cut)
        metrics['Recall@%d' % k] = \
            (hits.sum(1) / torch.min(torch.Tensor([k]).to(
                labels.device), labels.sum(1).float())).mean().cpu().item()
        
        metrics['MRR@%d' % k] = \
            (hits / torch.arange(1, k+1).unsqueeze(0).to(
                labels.device)).sum(1).mean().cpu().item()

        position = torch.arange(2, 2+k)
        weights = 1 / torch.log2(position.float())
        dcg = (hits * weights.to(hits.device)).sum(1)
        idcg = torch.Tensor([weights[:min(int(n), k)].sum()
                             for n in answer_count]).to(dcg.device)
        ndcg = (dcg / idcg).mean()
        metrics['NDCG@%d' % k] = ndcg.cpu().item()

    return metrics

class AverageMeterSet(object):
    def __init__(self, meters=None):
        self.meters = meters if meters else {}

    def __getitem__(self, key):
        if key not in self.meters:
            meter = AverageMeter()
            meter.update(0)
            return meter
        return self.meters[key]

    def update(self, name, value, n=1):
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, format_string='{}'):
        return {format_string.format(name): meter.val for name, meter in self.meters.items()}

    def averages(self, format_string='{}'):
        return {format_string.format(name): meter.avg for name, meter in self.meters.items()}

    def sums(self, format_string='{}'):
        return {format_string.format(name): meter.sum for name, meter in self.meters.items()}

    def counts(self, format_string='{}'):
        return {format_string.format(name): meter.count for name, meter in self.meters.items()}

class AverageMeter(object):
    def __init__(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
        
    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
        
    def update(self, val, n=1):   
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count
        
    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)

'''
Logger Parts
'''

class AbstractBaseLogger(metaclass=ABCMeta):
    @abstractmethod
    def log(self, *args, **kwargs):
        raise NotImplementedError

    def complete(self, *args, **kwargs):
        pass

class MetricGraphPrinter(AbstractBaseLogger):
    def __init__(self, key, graph_name, group_name):
        self.key = key
        self.graph_label = graph_name
        self.group_name = group_name
        
    def log(self, writer, *args, **kwargs):
        if self.key in kwargs:
            writer.add_scalar(self.group_name+'/'+ self.graph_label, kwargs[self.key], kwargs['accum_iter'])
        else:
            print('Metric {} not found...'.format(self.key))

    def complete(self, writer, *args, **kwargs):
        self.log(writer, *args, **kwargs)

class LoggerService(object):
    def __init__(self, args, writer, val_loggers, test_loggers):
        self.args = args
        self.writer = writer
        self.val_loggers = val_loggers if val_loggers else []
        self.test_loggers = test_loggers if test_loggers else []

    def complete(self):
        self.writer.close()

    def log_val(self, log_data):
        criteria_met = False
        for logger in self.val_loggers:
            logger.log(self.writer, **log_data)
            if self.args.early_stopping and isinstance(logger, BestModelLogger):
                criteria_met = logger.patience_counter >= self.args.early_stopping_patience
        return criteria_met
    
    def log_test(self, log_data):
        for logger in self.test_loggers:
            logger.log(self.writer, **log_data)


class RecentModelLogger(AbstractBaseLogger):
    def __init__(self, args, checkpoint_path, filename='checkpoint-recent.pth'):
        self.args = args
        self.checkpoint_path = checkpoint_path
        if not os.path.exists(self.checkpoint_path):
            self.checkpoint_path.mkdir(parents=True)
        self.recent_epoch = None
        self.filename = filename

    def log(self, *args, **kwargs):
        epoch = kwargs['epoch']

        if self.recent_epoch != epoch:
            self.recent_epoch = epoch
            state_dict = kwargs['state_dict']
            state_dict['epoch'] = kwargs['epoch']
            save_state_dict(state_dict, self.checkpoint_path, self.filename)

    def complete(self, *args, **kwargs):
        save_state_dict(kwargs['state_dict'],
                        self.checkpoint_path, self.filename + '.final')

class BestModelLogger(AbstractBaseLogger):
    def __init__(self, args, checkpoint_path, metric_key, filename='best_acc_model.pth'):
        self.args = args
        self.checkpoint_path = checkpoint_path
        if not os.path.exists(self.checkpoint_path):
            self.checkpoint_path.mkdir(parents=True)

        self.best_metric = 0.
        self.metric_key = metric_key
        self.filename = filename
        self.patience_counter = 0

    def log(self, *args, **kwargs):
        current_metric = kwargs[self.metric_key]
        if self.best_metric < current_metric:  # assumes the higher the better
            print("Update Best {} Model at {}".format(
                self.metric_key, kwargs['epoch']))
            self.best_metric = current_metric
            save_state_dict(kwargs['state_dict'],
                            self.checkpoint_path, self.filename)
            if self.args.early_stopping:
                self.patience_counter = 0
        elif self.args.early_stopping:
            self.patience_counter += 1
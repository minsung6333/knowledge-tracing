# from model import *
# from config import *
# from .utils import *
# from .loggers import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

import joblib
import json
import numpy as np
from abc import *
from pathlib import Path
from collections import OrderedDict
from utils import *

class BaseTrainer(metaclass=ABCMeta):
    def __init__(self, args, model, train_loader, val_loader, test_loader, infer_loader, export_root='./'):

        self.STATE_DICT_KEY = 'model_state_dict'
        self.OPTIMIZER_STATE_DICT_KEY = 'optimizer_state_dict'
        self.export_root = export_root
        if 'temp_dir' not in os.listdir(self.export_root):
            os.makedirs('./temp_dir')
            os.makedirs('./temp_dir/models')
        self.temp_root = os.path.join(self.export_root, 'temp_dir')
        
        self.args = args
        self.device = args.device
        self.model = model.to(self.device)
        self.num_epochs = args.num_epochs
        self.metric_ks = args.metric_ks
        self.best_metric = args.best_metric
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.infer_loader = infer_loader # D225374
        self.optimizer = self._create_optimizer()
        if args.enable_lr_schedule:
            if args.enable_lr_warmup:
                self.lr_scheduler = self.get_linear_schedule_with_warmup(
                    self.optimizer, args.warmup_steps, len(self.train_loader) * self.num_epochs
                )
            else:
                self.lr_scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=args.decay_step, gamma=args.gamma
                )
        
        # Logger Setting
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(
            log_dir=self.temp_root #,
            # comment=self.args.model_code+'_'+self.args.dataset_code,
        )
        self.val_loggers, self.test_loggers = self._create_loggers()
        self.logger_service = LoggerService(
            self.args, writer, self.val_loggers, self.test_loggers)
        
        print(args)
        print('Total parameters:', sum(p.numel() for p in model.parameters()))
        print('Encoder parameters:', sum(p.numel() for n, p in model.named_parameters() \
                                         if 'embedding' not in n))
                        
    def train(self):
        accum_iter = 0
        self.exit_training = self.validate(0, accum_iter)
        for epoch in range(self.num_epochs):
            accum_iter = self.train_one_epoch(epoch, accum_iter)
            if self.args.val_strategy == 'epoch':
                self.exit_training = self.validate(epoch, accum_iter)  # val after every epoch
            if self.exit_training:
                print('Early stopping triggered. Exit training')
                break
        self.logger_service.complete()

    def train_one_epoch(self, epoch, accum_iter):
        
        average_meter_set = AverageMeterSet()
        for batch_idx, batch in enumerate(self.train_loader):
            self.model.train()
            batch = self.to_device(batch)
            self.optimizer.zero_grad()
            loss = self.calculate_loss(batch)
            loss.backward()
            self.clip_gradients(self.args.max_grad_norm)
            self.optimizer.step()
            if self.args.enable_lr_schedule:
                self.lr_scheduler.step()

            accum_iter += 1
            if self.args.val_strategy == 'iteration' and accum_iter % self.args.val_iterations == 0:
                self.exit_training = self.validate(epoch, accum_iter)  # val after certain iterations
                if self.exit_training: break

        return accum_iter

    def validate(self, epoch, accum_iter):
        self.model.eval()
        average_meter_set = AverageMeterSet()
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                batch = self.to_device(batch)
                metrics = self.calculate_metrics(batch)
                self._update_meter_set(average_meter_set, metrics)

            log_data = {
                'state_dict': (self._create_state_dict()),
                'epoch': epoch+1,
                'accum_iter': accum_iter,
            }
            log_data.update(average_meter_set.averages())
        
        print(average_meter_set.averages())
        return self.logger_service.log_val(log_data)  # early stopping

    def test(self, epoch=-1, accum_iter=-1):
        print('******************** Testing Best Model ********************')
        best_model_dict = torch.load(os.path.join(
            self.temp_root, 'models', 'best_acc_model.pth')).get(self.STATE_DICT_KEY)
        self.model.load_state_dict(best_model_dict)
        self.model.eval()

        average_meter_set = AverageMeterSet()
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                batch = self.to_device(batch)
                metrics = self.calculate_metrics(batch)
                self._update_meter_set(average_meter_set, metrics)
                
            log_data = {
                'state_dict': (self._create_state_dict()),
                'epoch': epoch+1,
                'accum_iter': accum_iter,
            }
            average_metrics = average_meter_set.averages()
            log_data.update(average_metrics)
            self.logger_service.log_test(log_data)

            print('******************** Testing Metrics ********************')
            print(average_metrics)
            with open(os.path.join(self.temp_root, 'test_metrics.json'), 'w') as f:
                json.dump(average_metrics, f, indent=4)
        
        return average_metrics
    
    def infer(self): # D225374
        print('******************** Inference With Best Model ********************')
        best_model_dict = torch.load(os.path.join(
            self.temp_root, 'models', 'best_acc_model.pth')).get(self.STATE_DICT_KEY)
        self.model.load_state_dict(best_model_dict)
        self.model.eval()
        score_dict = {}
     
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.infer_loader):
                batch = batch.to(self.args.device)
                scores = self.calculate_prediction(batch)
                score_dict[batch_idx] = scores
        
        self.inference_result = score_dict
                
    # Not Needed To Be Modified
    def to_device(self, batch):
        return [x.to(self.device) for x in batch]

    # Not Needed To Be Modified
    @abstractmethod
    def calculate_prediction(self, batch):
        pass
    
    # Not Needed To Be Modified
    @abstractmethod
    def calculate_loss(self, batch):
        pass

    # Not Needed To Be Modified
    @abstractmethod
    def calculate_metrics(self, batch):
        pass

    # Not Needed To Be Modified
    def clip_gradients(self, limit=1.0):
        nn.utils.clip_grad_norm_(self.model.parameters(), limit)

    # Not Needed To Be Modified
    def _update_meter_set(self, meter_set, metrics):
        for k, v in metrics.items():
            meter_set.update(k, v)

    # Not Needed To Be Modified
    def _create_optimizer(self):
        args = self.args
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'layer_norm']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay,
            },
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.},
        ]
        if args.optimizer.lower() == 'adamw':
            return optim.AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
        elif args.optimizer.lower() == 'adam':
            return optim.Adam(optimizer_grouped_parameters, lr=args.lr, weight_decay=args.weight_decay)
        else:
            raise NotImplementedError

    # Not Needed To Be Modified
    def get_linear_schedule_with_warmup(self, optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
        
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            )

        return LambdaLR(optimizer, lr_lambda, last_epoch)

    def _create_loggers(self):
        root = Path(self.temp_root)
        model_checkpoint = root.joinpath('models')

        val_loggers, test_loggers = [], []
        for k in self.metric_ks:
            val_loggers.append(
                MetricGraphPrinter(key='Recall@%d' % k, graph_name='Recall@%d' % k, group_name='Validation'))
            val_loggers.append(
                MetricGraphPrinter(key='NDCG@%d' % k, graph_name='NDCG@%d' % k, group_name='Validation'))
            val_loggers.append(
                MetricGraphPrinter(key='MRR@%d' % k, graph_name='MRR@%d' % k, group_name='Validation'))

        val_loggers.append(RecentModelLogger(self.args, model_checkpoint))
        val_loggers.append(BestModelLogger(self.args, model_checkpoint, metric_key=self.best_metric))

        for k in self.metric_ks:
            test_loggers.append(
                MetricGraphPrinter(key='Recall@%d' % k, graph_name='Recall@%d' % k, group_name='Test'))
            test_loggers.append(
                MetricGraphPrinter(key='NDCG@%d' % k, graph_name='NDCG@%d' % k, group_name='Test'))
            test_loggers.append(
                MetricGraphPrinter(key='MRR@%d' % k, graph_name='MRR@%d' % k, group_name='Test'))

        return val_loggers, test_loggers

    def _create_state_dict(self):
        return {self.STATE_DICT_KEY: self.model.state_dict(), self.OPTIMIZER_STATE_DICT_KEY: self.optimizer.state_dict()}

# modifed done
class LRUTrainer(BaseTrainer):
    
    # Not Needed To Be Modified
    def __init__(self, args, model, train_loader, val_loader, test_loader, infer_loader, export_root='./'):
        super().__init__(args, model, train_loader, val_loader, test_loader, infer_loader, export_root)
        self.ce = nn.CrossEntropyLoss(ignore_index=0) # paper : nll / torch.nn.ce : log_softmax+nll

    # Not Needed To Be Modified
    # modified by ssgds (D225374) : 수정완료
    def calculate_loss(self, batch):
        seqs, labels = batch
        logits = self.model(seqs)[0]
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)
        loss = self.ce(logits, labels)
        return loss

    # Not Needed To Be Modified
    # modified by ssgds (D225374) : 수정완료
    def calculate_metrics(self, batch):
        seqs, labels = batch
        scores = self.model(seqs)[0][:, -1, :]
        B, L = seqs.shape
        for i in range(L):
            scores[torch.arange(scores.size(0)), seqs[:, i]] = -1e9
        scores[:, 0] = -1e9  # padding
        metrics = absolute_recall_mrr_ndcg_for_ks(scores, labels.view(-1), self.metric_ks) # called from utils
        return metrics
    
    # modified by ssgds (D225374) : 추가완료
    def calculate_prediction(self, batch):
        seqs = batch
        scores = self.model(seqs)[0][:, -1, :]
        B, L = seqs.shape
        for i in range(L):
            scores[torch.arange(scores.size(0)), seqs[:, i]] = -1e9
        scores[:, 0] = -1e9  # padding
        return scores
        
    
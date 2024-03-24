class CONFIGS: # need to add 'reset_conifg' method
    def __init__(self):
        self.train_batch_size = 64 
        self.val_batch_size = 128
        self.test_batch_size = 128
        self.infer_batch_size = 128
        self.num_workers = 8 
        self.sliding_window_size = 1
        # self.negative_sample_size = 50 
        self.device = 'cuda:0'
        self.num_epochs = 100
        self.optimizer = 'AdamW'
        self.weight_decay = 0.01
        self.adam_epsilon = 1e-9
        self.momentum = None
        self.lr = 0.001
        self.max_grad_norm = 5.0
        self.enable_lr_schedule = False
        self.decay_step = 10000
        self.gamma = 0.1
        self.enable_lr_warmup = False
        self.warmup_steps = 100
        self.val_strategy = 'iteration'
        self.val_iterations = 1000
        self.early_stopping = True
        self.early_stopping_patience = 10
        self.metric_ks = [1, 5, 10, 20]
        self.best_metric = 'Recall@20'
        self.bert_max_len = 20
        self.bert_hidden_units = 64
        self.bert_num_blocks = 2
        self.bert_num_heads = 2
        self.bert_head_size = None
        self.bert_dropout = 0.2
        self.bert_attn_dropout = 0.2
        self.bert_mask_prob = 0.2
        
    def update_num_items(self, num_items):
        self.num_items = num_items
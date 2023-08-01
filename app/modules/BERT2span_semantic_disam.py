import os
import numpy as np
import random
import json
from typing import List, Dict, Tuple, Callable, Any, Optional, Union, Iterable

import pytorch_lightning as pl

from torch import nn
import torch.nn.functional as F
from torch import nn
from transformers import (AdamW, WEIGHTS_NAME, get_linear_schedule_with_warmup)

class BERT2span(pl.LightningModule):

    def __init__(self, configs, bert_model, tokenizer, vocab_type='wordpiece', freeze=False, add_kldiv= False):

        super().__init__()
        config_dict = self._load_configs(configs)

        # Tokenizer has added drug tokens, task codes ([NER], [POS], [DEP])
        self.tokenizer = tokenizer
        self.vocab_dict = {i:v for v,i in tokenizer.vocab.items()}
        self.vocab_type = vocab_type
        self.freeze = freeze

        # BERT is initialized with pre-trained model and resize embeddings (the embeddings of new added tokens are initialized randomly)
        self.encoder = bert_model
        self.mlm = False

        self._reset_binary_classifer(self.encoder.config.hidden_size)
     
        
        self.bce_loss = nn.BCEWithLogitsLoss(reduction = "mean")
        if add_kldiv:
            self.kldiv_loss =  nn.KLDivLoss(reduction="mean")
        else:
            self.kldiv_loss = None

    
    def _reset_binary_classifer(self, input_size):
        self.binary_classifier = nn.Linear(input_size, 1, bias=True)
        
    
    def _load_configs(self, config):
        """
        Load the data, model, training, testing configs from one yaml file
        :param config_file: 
        :return: 
        """
        #config = load_config(config_file)
        self.data_config = config['data']
        self.model_config = config['model']
        self.train_config = config['train']
        self.test_config = config['test']
        self.model_dir = self.train_config['save_path']
        #self.label_config = config['tokens']
        return config
    
    
    def forward(self, input_ids, attention_mask, targets = None, definition_hidden_mean= None):
        """
        label_word: string, e.g. disorders
        """
        
        outputs = self.encoder(input_ids.to(self.encoder.device), attention_mask.to(self.encoder.device), output_hidden_states = True)
        last_hidden_output = outputs.hidden_states[-1]

        emission = self.binary_classifier(last_hidden_output)
        
        if targets is not None:

            if len(targets.shape) < 3:
                targets = targets.unsqueeze(-1)
            loss = self.bce_loss(emission, targets)
            
            if self.kldiv_loss and definition_hidden_mean is not None:
                last_hidden_output_mask = last_hidden_output * targets
                last_hidden_output_mean = F.log_softmax(last_hidden_output_mask.mean(1), dim=-1)
                loss += self.kldiv_loss(last_hidden_output_mean, F.softmax(definition_hidden_mean, dim=-1))
            return loss
        else:
            prediction = F.sigmoid(emission)
            return prediction

    def training_step(self, src_batch, batch_idx):
        """
        :param batch:
        :param batch_idx:
        :return:
        """
     
        loss = self(**src_batch)
        self.log('train_loss', loss)

        return loss

    def validation_step(self, src_batch, batch_idx):

        loss = self(**src_batch)
        self.log('val_loss', loss)
    
    def test_step(self, src_batch, batch_idx):
        
        #src_batch, sorted_idx = self._tokenize(batch)

        loss = self(**src_batch)
        self.log('test_loss', loss)
   
    def configure_optimizers(self):
        """
        :param config: training configs
        :return:
        """
        weight_decay = self.train_config.get("weight_decay", 0.3)
        no_decay = ["bias", "LayerNorm.weight"]
        
        if self.freeze:
            optimizer_grouped_parameters = [
                {"params": [p for n, p in  self.binary_classifier.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,}, 
             
                {"params": [p for n, p in  self.binary_classifier.named_parameters() if  any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,}]


        else:
            optimizer_grouped_parameters = [
                {"params": [p for n, p in self.encoder.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,},
                {"params": [p for n, p in self.binary_classifier.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,}, 
                
                {"params": [p for n, p in self.encoder.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,},
                {"params": [p for n, p in self.binary_classifier.named_parameters() if  any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,}, 
               ]

        
        #optimizer = build_optimizer(self.train_config, optimizer_grouped_parameters)
        lr = float(self.train_config['learning_rate'])
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
        print('optimizer lr ', optimizer.param_groups[0]['lr'])
        #scheduler_mode = 'min'
        #scheduler, scheduler_step_at = build_scheduler(self.train_config, optimizer, scheduler_mode, self.model.decoder.config.hidden_size)
        #scheduler_config = {"scheduler": scheduler, "interval": scheduler_step_at, "frequency": 1}
        return optimizer

    def __repr__(self) -> str:
        """
        String representation: a description of encoder, decoder and embeddings
        :return: string representation
        """
        return "%s(\n" \
               "\tencoder=%s,\n" \
               "\tcls=%s )"% (self.__class__.__name__, self.encoder,
                                  self.binary_classifier)


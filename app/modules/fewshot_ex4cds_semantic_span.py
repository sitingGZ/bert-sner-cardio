#from lib2to3.pgen2 import token
import os
import json
from copy import deepcopy
from re import L
import pickle
from typing import Dict, List, Tuple
from random import shuffle
import numpy as np
import collections

import torch
from torch.utils.data import DataLoader as batch_gen
from helpers import  load_config, set_seed, average_checkpoints

from pytorch_lightning import Trainer, callbacks
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from transformers import AutoConfig, AutoTokenizer, AutoModelForMaskedLM
from BERT2span_semantic_disam import BERT2span

#from AdapterBertMLM import AdapterBertForMaskedLM

ex4cds_labels_terms = {'Condition':'Zeichen oder Symptom', 
                 'DiagLab':'Diagnostisch und Laborverfahren', 
                 'LabValues':'Klinisches Attribut', 
                 'HealthState':'Gesunder Zustand', 
                 'Measure':'Quantitatives Konzept', 
                 'Medication': 'Pharmakologische Substanz',
                 'Process':'Physiologische Funktion', 
                 'TimeInfo':'Zeitliches Konzept'}

ex4cds_labels = list(ex4cds_labels_terms.keys())

from helpers import remove_umlaut

def tokenize_text(text):
    tokens = {}
    for l in text:
        #print(begin)
        tokens[int(l[0])] = {'id': int(l[1]), 'text': l[2]}
        #begin += len(w)+1
        
    return tokens

def ex4cds_token_and_tag_pair(text, anns, exception = ['Temporal', 'Other', 'Conclusion']):
    anns_idx = {}
    for l in anns:
        if l[0][0] == 'T':
            label = l[1].split()
            if label[0] not in exception:
                text_split = l[2].split()
                current_begin = int(label[1])
                for t in text_split:
                    current_end = current_begin + len(t)
                    anns_idx[current_begin] = {'begin': current_begin, 'end': current_end, 'label': label[0], 'text': t}
                    current_begin = current_end + 1
                    
    tokens = tokenize_text(text)
    words = [v['text'] for i, v in tokens.items()]
    tags = ['O'] * len(tokens)
    for b,ann in anns_idx.items():
        if ann['begin'] in tokens:
            idx = tokens[ann['begin']]['id']
            tags[idx] = ann['label']
    return words, tags

def tokenize_and_tagging_ex4cds(seq_label_pair, tokenizer, label_words):
    words = seq_label_pair[0]
    orig_tags = seq_label_pair[1]
    #current_labels = [tag[2:] for tag in tags if tag != 'O']
    #targets = {label: [] for label in current_labels}
    label_tokens = {t: tokenizer.encode(label) for t, label  in label_words.items()}
    
    targets = {}
    
    for t, input_tokens in label_tokens.items():
 
        label_has_tags = False
        label_tags_none = [0] * len(input_tokens)
        label_tags_true  = [0]
        label_tags_true += [1]* (len(input_tokens)-2)
        label_tags_true += [0]
                    
        subtags = []
        for i, w in enumerate(words):
            encoded = tokenizer.encode(w)[1:-1]
            input_tokens += encoded
            #if orig_tags[i] != 'O':
            if orig_tags[i] == t:
                subtags.extend([1] * len(encoded))
                label_has_tags = True
                #else:
                    #subtags.extend([0] * len(encoded))
            else:
                subtags.extend([0] * len(encoded))
        
        if label_has_tags:
            tags = label_tags_true + subtags
        else:
            tags = label_tags_none + subtags
            
        input_tokens.append(tokenizer.sep_token_id)
        tags.append(0)
        #span_targets.append(tags[:tokenizer.model_max_length])
        #input_tokens.append(tokens[:tokenizer.model_max_length])                    
        targets[t] = (input_tokens, tags)
    return targets
                
def _padding(new_batch, pad_token_id, training = False):
        ''' Pads to the longest sample
            batch[idx][0]: length of each encoded sequence 
            batch[idx][1]: the encoded sequence 
            batch[idx][2]: sequence of tags
            
        '''
        get_element = lambda x: [sample[x] for sample in new_batch]
        # Each 
        seq_len = get_element(0)
        #print(seq_len)
        maxlen = np.array(seq_len).max()
        
        do_pad_ids = lambda x, seqlen, padding: [sample[x] + [padding] * (seqlen - len(sample[x])) for sample in new_batch] # 0: <pad>
        tok_ids = do_pad_ids(1, maxlen, pad_token_id)
        attn_mask = [[(i != pad_token_id) for i in ids] for ids in tok_ids] 
        
        # sort the index, attn mask and labels on token length
        token_ids = get_element(1)
        token_ids_len = torch.LongTensor(list(map(len, token_ids)))
        _, sorted_idx = token_ids_len.sort(0, descending=True)

        tok_ids = torch.LongTensor(tok_ids)[sorted_idx]
        attn_mask = torch.LongTensor(attn_mask)[sorted_idx]

        if training: 
            do_pad_labels = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in new_batch]
            labels = do_pad_labels(2, maxlen)
           
            #attn_mask = [[(i != self.tag2id['<pad>']) for i in ids] for ids in label]
            labels = torch.Tensor(labels)[sorted_idx]
            #definition_hidden_mean_tensor = torch.cat([new_batch[i][3] for i in range(len(new_batch))], dim = 0)[sorted_idx]
        else:
            labels = None
            #definition_hidden_mean_tensor = None
        return tok_ids, attn_mask, labels,  sorted_idx.cpu().numpy()

class TokenizationDataset(torch.utils.data.Dataset):
    """Construct data set for dataloader"""

    def __init__(self, input_ids:torch.Tensor, attention_mask: torch.Tensor, targets: torch.Tensor):
        
        """
        
        """
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.targets = targets
        
    def __getitem__(self, index:int)->Dict:
        """
        :param index : int
        """
        result = {'input_ids':  self.input_ids[index], 'attention_mask': self.attention_mask[index], 'targets':self.targets[index]}
        return result
    
    def __len__(self):
        return len(self.input_ids)

def main(config_file):
    # Prepare configurations
    configs = load_config(config_file)
    seeds= configs['train'].get('random_seeds', [42])
    
    # language of pre-trained BERT: english, german or multilingual
    batch_size = configs['train']['batch_size']
    gpus = 1
    max_epochs = configs['train']['epochs']
                                  
    # source path to raw text, target path to inserted NER annotations
    # data sets of n2c2 and muchmore
    dataset_name = configs['data']['name']
    language = configs['data']['language']

    bert_languages = configs['data']['bert_languages']

    update = True
    ex4cds_path = configs['data']['dataset'][dataset_name]['data_path']
    shots = configs['data']['dataset'][dataset_name]['shots']
    #files =  os.listdir(conll_train_path)

    
    seeds = [42, 99]
    #print('train and valid data amount', len(train_data), len(valid_data) )
    for seed in seeds:
        torch.cuda.empty_cache()
        set_seed(seed=seed)
        for bert_language, params in bert_languages.items():
            base_names = configs['model']['encoder'][bert_language]
            freeze = configs['train']['freeze']
            add_kldiv = configs['train']['add_kldiv']
            tokenizer_path = configs['model']['tokenizer'][bert_language]
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            batch_size = params['batch_size']
            #bronco_pairs = []
            train_data = []
            valid_data = []
    
    
            files = [f.split('.')[0] for f in os.listdir(ex4cds_path) if '.' in f]
            print(len(files))
            split_train_valid = int(len(files) * 0.7)
            for i, f in enumerate(files):
                    #print(os.path.join(ex4cds_path, f + '.txt'))
                    text = [l.strip().split('\t') for l in open(os.path.join(ex4cds_path, f + '.txt'), encoding='latin-1')]
                    annos = [l.strip().split('\t') for l in open(os.path.join(ex4cds_path, f+ '.ann'), encoding='latin-1')]
                    #print(text, annos)
                    words,tags = ex4cds_token_and_tag_pair(text,annos)
                    #print(words, tags)
                    #if len(words) < 1:
                    #    print(text, annos, words, tags)
                    current_sample = tokenize_and_tagging_ex4cds((words,tags), tokenizer=tokenizer, label_words=ex4cds_labels_terms)
                    if i < split_train_valid:
                        train_data.append(current_sample)
                    else:
                        valid_data.append((current_sample))

            print('valid_data example ', valid_data[:2])
            ckpt_paths = []
                #for r in range(3):
            valid_batches = []

            for idx , valid_input_tag in enumerate(valid_data):
                for l, input_tag in valid_input_tag.items():
                    valid_batches.append((len(input_tag[0]), input_tag[0], input_tag[1]))

             
            if 'pre_checkpoints' in configs['train']:
                save_dir = configs['train']['pre_checkpoints'][language]
                r = 1
                for length in [40, 80]:
                    ckpt_path = "bert2span_seed_42_round_{}_val_batch_{}".format(r, length)
                    ckpt_dir = os.path.join(save_dir, ckpt_path)
                        #ckpt_dir = os.path.join(save_path, ckpt_path)
                    ckpts = os.listdir(ckpt_dir)
                    ckpt_paths.extend([os.path.join(ckpt_dir, ckpt) for ckpt in ckpts])
                    print(ckpt_paths)
                #best_ckpt = find_best_checkpoint(os.listdir(ckpt_dir))
                
                #set_seed(seed=42)
            if len(ckpt_paths) > 0:
                average_state = average_checkpoints(ckpt_paths)
            
            for i in range(3):
                for name, base_name in base_names.items():
            
                    bertMLM = AutoModelForMaskedLM.from_pretrained(base_name)
        
                    bert2span = BERT2span(configs, bertMLM, tokenizer, freeze=freeze, add_kldiv=False)
                
                    save_path = configs['train']['save_path'].format(language, dataset_name, add_kldiv,freeze, name)
                
                    #save_path = model_path.format(mlm_pretrained, mlm_training, name + '_update_encoder_{}_we_{}'.format(update_encoder, update_we))
                    if not os.path.exists(save_path):
                            os.mkdir(save_path)

     

                    tok_ids, attn_mask, targets,  sorted_indices  = _padding(valid_batches, pad_token_id=tokenizer.pad_token_id, training=True)
                            
                    current_v_dataset = TokenizationDataset(input_ids= tok_ids, targets = targets, attention_mask = attn_mask)
                    valid_dataset = batch_gen(current_v_dataset, batch_size, num_workers=2)
             
                
                    if 'no_pretrained' not in save_path:
                        bert2span.load_state_dict(average_state['state_dict'])
                        print('loaded pretrained checkpoint')
                        
                    shuffle(train_data)
                    for shot in shots:
                        #current_train_ids = train_level_ids[stage]
                        #current_valid_ids = valid_level_ids[stage]
                        r = 'seed_{}_round_{}_shot_{}'.format(seed,i, shot)
                        #tokens = []
                        #tags = []
                        #label_masks = []
                    
                        train_batches = []
                        
                        for tidx, train_input_tags in enumerate(train_data[:shot]):
                            #if len(train_data[tidx]) > 1:
                            for l, train_input_tag in train_input_tags.items():
                                train_batches.append((len(train_input_tag[0]), train_input_tag[0], train_input_tag[1]))

                    
                        tok_ids, attn_mask, targets,  sorted_indices  = _padding(train_batches, pad_token_id=tokenizer.pad_token_id, training=True)
                        current_train_dataset = TokenizationDataset(input_ids= tok_ids, targets = targets, attention_mask = attn_mask)
                        train_dataset = batch_gen(current_train_dataset, batch_size, num_workers=2)
                            
                    
                        cur_model_dir = os.path.join(save_path, 'bert2span_{}'.format(r))
                        checkpoint_callback = callbacks.ModelCheckpoint(monitor='val_loss', dirpath=cur_model_dir, mode = 'min',
                                                    filename='bert2span-{epoch:02d}-{val_loss:.5f}', save_top_k=2)
                        trainer = Trainer(gpus=gpus, gradient_clip_val = 1.0, stochastic_weight_avg=True, max_epochs=max_epochs,callbacks=checkpoint_callback, precision=16)      
                    
                        trainer.fit(bert2span, train_dataset, valid_dataset)
                                #print('Best model path of task {}'.format(task + round), checkpoint_callback.best_model_path)
                        torch.cuda.empty_cache()


if __name__ == '__main__':
    import sys
    #config_path = 'configs/transformer_config.yaml'
    #main(config_path)
    assert len(sys.argv) >= 2, "The path to the config file must be given as argument!"
    main(sys.argv[1])

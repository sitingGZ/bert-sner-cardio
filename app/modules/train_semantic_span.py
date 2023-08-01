#from lib2to3.pgen2 import token
import os
import json
from copy import deepcopy
from re import L
import pickle
from typing import Dict, List, Tuple
from random import shuffle
import numpy as np


import torch
from torch.utils.data import DataLoader as batch_gen
from helpers import find_best_checkpoint, load_config, set_seed

from pytorch_lightning import Trainer, callbacks
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from transformers import AutoConfig, AutoTokenizer, AutoModelForMaskedLM
from BERT2span_semantic_disam import BERT2span

#from AdapterBertMLM import AdapterBertForMaskedLM

labels_id_german = {
 1: 'Aktivitaeten und Verhaltensweisen',
 11: 'Anatomie',
 9: 'Chemikalien & Drogen',
 13: 'Konzepte & Ideen',
 5: 'Geraete',
 4: 'Stoerungen',
 8: 'Gene und molekulare Sequenzen',
 6: 'Geografische Gebiete',
 14: 'Lebewesen',
 10: 'Objekte',
 12: 'Berufe',
 2: 'Organisationen',
 15: 'Phaenomene',
 7: 'Physiologie',
 3: 'Verfahren'}

def get_label_to_id(child_parents_dict):
    labels = {0: 'T000'}
    for i, t in enumerate(child_parents_dict.keys()):
        labels[i+1] = t
    labels[len(labels)] = 'T999'
    label_to_id = {l: i for i, l in labels.items()}
    #len(label_to_id)
    return labels, label_to_id

def retrieve_level_sent_ids(dataset):
    # All levels [2, 3, 4, 5, 6, 7, 8], the lower level are parent terms
    level_sent_ids = {i: [] for i in range(2,9)}
    # Each sample in dataset
    for i, input_word_tags in enumerate(dataset):
        max_level = 1
        # Each token in current sample
        for w,d in input_word_tags.items():
            # Find out the deepest level
            max_level = max(max_level, max(d['tui'].keys()))
            
        for k, l in level_sent_ids.items():
            # Add the current sample to the lower levels
            if k<=max_level:
                l.append(i)
    return level_sent_ids

def _term_definition_hiddens(tokenizer, bert, term_def_dict):
    term_def_text = {k: '[SEP]'.join(vs) for k, vs in term_def_dict.items()}
    term_def_tokens = {k: tokenizer([text], return_tensors = 'pt') for k, text in term_def_text.items()}
    term_def_hidden_mean = {k: bert(**inputs, output_hidden_states = True).hidden_states[-1].mean(1) for k, inputs in term_def_tokens.items()}
    return term_def_hidden_mean

def get_semantic_group_annos(input_word_tag):
    """
    input_word_tag = {'w1': {'text': 'Zusammenfassung', 'pos': 'NN', 'tui': {0: 'T000'}},
      'w2': {'text': ':', 'pos': 'PUNCT', 'tui': {0: 'T000'}},
      'w3': {'text': 'Die', 'pos': 'ART', 'tui': {0: 'T000'}},
      'w4': {'text': 'Enteritis',
       'pos': 'NN',
       'tui': {0: 'T000', 6: 'T047', 4: 'T184'}}}

    tui_to_id: tui id to semantic id
    """
    words = [d['text'] for w, d in input_word_tag.items()]
    tags = {}
    
    for w, d in input_word_tag.items():
        
        if len(d['tui']) > 1:
            idx = int(w[1:])-1
            for k, t in d['tui'].items():
                if k > 0 : 
                    if t not in tags:
                        tags[t] = [0] * len(input_word_tag)
                    tags[t][idx] = 1
            
    return words, tags


def tokenize_and_tagging(tokenizer, input_word_tag: dict, labels:dict, term_def_hidden_mean:dict):
    """
    Args: 
        input_word_tag 
        label_to_id 
    """
    current_words, current_tag = get_semantic_group_annos(input_word_tag)
    label_tokens = {t: tokenizer.encode(labels[t]) for t in current_tag if t in labels if t != 'T000'}
    
    # 1 for the position of prefix tokens of interest, they should be tagged
    
    #init_tags = [0] * len(label_to_id)
    #init_tags[-1] = 1
    #tags = [init_tags]
    span_targets = []
    definition_mean_targets = []
    input_tokens = []
    
    for t, tokens in label_tokens.items():
        #sub_tokens = []
        sub_tags = []
        if t in term_def_hidden_mean:
            label_has_tags = False
            label_tags_none = [0] * len(tokens)
            label_tags_true  = [0]
            label_tags_true += [1]* (len(tokens)-2)
            label_tags_true += [0]
                    
            for i, word in enumerate(current_words):
                sub_tokens = tokenizer.encode(word)[1:-1]
                tokens.extend(sub_tokens)
                #sub_tags = [0] * len(sub_tokens)
                if current_tag[t][i] > 0:
                    sub_tags.extend([1] * len(sub_tokens))
                    label_has_tags = True
                else:
                    sub_tags.extend([0] * len(sub_tokens))
                    
                #sub_tags.extend(sub_tags)
            if label_has_tags:
                tags = label_tags_true + sub_tags
            else:
                tags = label_tags_none + sub_tags
                
            tokens.append(tokenizer.sep_token_id)
            tags.append(0)
            
            span_targets.append(tags[:tokenizer.model_max_length])
            input_tokens.append(tokens[:tokenizer.model_max_length])
            
            if label_has_tags:
                definition_mean_targets.append(term_def_hidden_mean[t])
            else:
                definition_mean_targets.append(term_def_hidden_mean['T000'])
                
    return input_tokens, span_targets, definition_mean_targets

def find_max_length(current_length, batch_lengths = [10, 20, 30, 40, 60, 80, 100]):
    min_gap = np.argmin([(l - current_length) if l >= current_length else current_length for l in batch_lengths])
    return batch_lengths[min_gap]

def _padding(new_batch, pad_token_id, training = False):
        ''' Pads to the longest sample
            batch[idx][0]: length of each encoded sequence 
            batch[idx][1]: the encoded sequence 
            batch[idx][2]: sequence of tags
            batch[idx][3]: batch of definition_mean_hidden
     
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
            
            definition_hidden_mean_tensor = torch.cat([new_batch[i][3] for i in range(len(new_batch))], dim = 0)[sorted_idx]
            
        else:
            labels = None
            definition_hidden_mean_tensor = None

        #org_tok_map = get_element(2)
        #sents = get_element(-1)
        return tok_ids, attn_mask, labels, definition_hidden_mean_tensor.detach(),  sorted_idx.cpu().numpy()


class TokenizationDataset(torch.utils.data.Dataset):
    """Construct data set for dataloader"""

    def __init__(self, input_ids:torch.Tensor, attention_mask: torch.Tensor, targets: torch.Tensor , definition_hidden_mean: torch.Tensor):
        
        """
        Source data sample: 
        input_word_tag = {'w1': {'text': 'Zusammenfassung', 'pos': 'NN', 'tui': {0: 'T000'}},
  'w2': {'text': ':', 'pos': 'PUNCT', 'tui': {0: 'T000'}},
  'w3': {'text': 'Die', 'pos': 'ART', 'tui': {0: 'T000'}},
  'w4': {'text': 'Enteritis',
   'pos': 'NN',
   'tui': {0: 'T000', 6: 'T047', 4: 'T184'}}}
    After tokenization and taggings:
        tokens: list of tokens
        tags: list of multiclass tags
        label_mask: mask subtokens and special tokens
        """
        
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.targets = targets
        self.definition_hidden_mean = definition_hidden_mean
        
    def __getitem__(self, index:int)->Dict:
        """
        :param index : int
        """
        result = {'input_ids':  self.input_ids[index], 'attention_mask': self.attention_mask[index], 'targets':self.targets[index], 'definition_hidden_mean': self.definition_hidden_mean[index]}
        return result
    
    def __len__(self):
        return len(self.input_ids)


def main(config_file):
    # Prepare configurations
    configs = load_config(config_file)
    seeds= configs['train'].get('random_seeds', [42, 99])
    
    # language of pre-trained BERT: english, german or multilingual
    batch_size = configs['train']['batch_size']
    gpus = 1
    max_epochs = configs['train']['epochs']
                                  
    # source path to raw text, target path to inserted NER annotations
    # data sets of n2c2 and muchmore
    dataset = configs['data']['dataset']
    language = configs['data']['language']

    bert_languages = configs['data']['bert_languages']

    # tasks: NER and MLM
    child_parents_jsfile = configs['data']['child_parents_jsfile']
    term_semantic_lines = [l.split('\t') for l in open(configs['data']['semantic_group_types'])]
    sty_definition_json = configs['data']['sty_definition_json'][language]
    #'/ds/text/iml_liang/muchmore/tui_sem_dict_2001_english_extended.json'
    term_def_dict = json.load(open(sty_definition_json))
    
    print(term_semantic_lines[:2])
    
    tui_to_id = {l[0]: int(l[1]) for l in term_semantic_lines}
    #tui_to_labels = {l[0]: l[2] for l in term_semantic_lines}
    tui_to_labels = {t:v[0] for t, v in term_def_dict.items()}
    
    
    #if language == 'german':
    #    tui_to_labels = {l:labels_id_german[tui_to_id[l]] for l in tui_to_labels}
    #label_to_id.update({'Padding':len(label_to_id)})
    #child_parents_dict = json.load(open(child_parents_jsfile))
    #labels, label_to_id = get_label_to_id(child_parents_dict)

    update = True
    train_path = configs['data']['train_source'].format(dataset, language)
    valid_path = configs['data']['valid_source'].format(dataset, language)
        
    print(dataset, train_path, valid_path)
    train_data = pickle.load(open(train_path, 'rb'))
    valid_data = pickle.load(open(valid_path, 'rb'))


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
        
            for name, base_name in base_names.items():
            
                bertMLM = AutoModelForMaskedLM.from_pretrained(base_name)
                definition_hidden_mean_dict =  _term_definition_hiddens(tokenizer, bertMLM, term_def_dict)
                  
                #bertMLM.bert.encoder.requires_grad = update
                bert2span = BERT2span(configs, bertMLM, tokenizer, freeze=freeze, add_kldiv=add_kldiv)
                
                
                if "pre_checkpoints" in configs['train']:
                        
                        ckpt_paths = configs['train']['pre_checkpoints']
                   
                        # Only update embeddings in step2
                        step2_path = deepcopy(ckpt_paths['step2']).format(name)
                        #step2_path = deepcopy(ckpt_paths['step2']).format( param[0], param[1], param[2])
                
                        step2_data_path = deepcopy(ckpt_paths['data_dir_step2']).format(seed)

                        ckpt_dir = os.path.join(step2_path, step2_data_path)
                        assert os.path.exists(ckpt_dir), "Check step2 model path {}".format(step2_path) 
                        assert len(os.listdir(ckpt_dir)) >=1, "Check ckpt_dir {}".format(ckpt_dir)
                        best_ckpt = find_best_checkpoint(os.listdir(ckpt_dir))
                        
                        best_state_dict = torch.load(os.path.join(ckpt_dir, best_ckpt)) 
                        bert2span.load_state_dict(best_state_dict['state_dict'])
                        #mlm_pretrained = True
                        mlm_pretrained = step2_path.split('/')[-1]
                #for update_encoder in updates:    
                # add sequence labels in to decoder vocabulary
                #tokenizer.add_tokens(new_tokens)
                save_path = configs['train']['save_path'].format(language, dataset, freeze, add_kldiv, name)
                    #save_path = model_path.format(mlm_pretrained, mlm_training, name + '_update_encoder_{}_we_{}'.format(update_encoder, update_we))
                if not os.path.exists(save_path):
                            os.mkdir(save_path)

                #print('total_stages tran and valie', train_level_ids.keys(), valid_level_ids.keys())            
                batch_lengths = [40, 80]
                valid_batches = {i: [] for i in batch_lengths}
                valid_batches[-1] = []
                for idx , valid_input_word_tag in enumerate(valid_data):
                        if len(valid_data[idx]) > 1:
                            tokens, tags, def_hidden_mean = tokenize_and_tagging(tokenizer, input_word_tag= valid_input_word_tag, labels=tui_to_labels, term_def_hidden_mean=definition_hidden_mean_dict)
                            
                            for j in range(len(tokens)):
                                if len(tokens[j]) <= batch_lengths[-1]:
                                    max_length = find_max_length(len(tokens[j]), batch_lengths)
                                
                                    valid_batches[max_length].append((len(tokens[j]), tokens[j], tags[j], def_hidden_mean[j]))
                                else:
                                    valid_batches[-1].append((len(tokens[j]), tokens[j], tags[j], def_hidden_mean[j]))

                mini_valid_batch_padded = {}
                for maxlen, v_batch in valid_batches.items():
                        if len(v_batch) > 0:

                            tok_ids, attn_mask, targets, definition_hidden_mean, sorted_indices  = _padding(v_batch, pad_token_id=tokenizer.pad_token_id, training=True)
                            
                            current_v_dataset = TokenizationDataset(input_ids= tok_ids, targets = targets, attention_mask = attn_mask, definition_hidden_mean=definition_hidden_mean)
                            mini_valid_batch_padded[maxlen] = batch_gen(current_v_dataset, batch_size, num_workers=2)
                            print('valid_batch size and amount ', maxlen, len(v_batch))
                        else:
                            print('no batch in current stage, current_maxlen and len of batch', stage, maxlen, len(v_batch) )

                  
                for stage in range(3):
                    #current_train_ids = train_level_ids[stage]
                    #current_valid_ids = valid_level_ids[stage]
                    r = 'seed_{}_round_{}'.format(seed,stage)
                    #tokens = []
                    #tags = []
                    #label_masks = []
                    train_batches = {i: [] for i in batch_lengths}
                    train_batches[-1] = []
                    shuffle(train_data)
                    for tidx, train_input_word_tag in enumerate(train_data):
                        if len(train_data[tidx]) > 1:
                            tokens, tags, def_hidden_mean = tokenize_and_tagging(tokenizer, input_word_tag= train_input_word_tag, labels=tui_to_labels, term_def_hidden_mean=definition_hidden_mean_dict)
                            
                            for j in range(len(tokens)):
                                if len(tokens[j]) <= batch_lengths[-1]:
                                    max_length = find_max_length(len(tokens[j]), batch_lengths)
                                
                                    train_batches[max_length].append((len(tokens[j]), tokens[j], tags[j], def_hidden_mean[j]))
                                else:
                                    train_batches[-1].append((len(tokens[j]), tokens[j], tags[j], def_hidden_mean[j]))

                    mini_train_batch_padded = {}
                    for maxlen, batch in train_batches.items():
                        if len(batch) > 0:
                            tok_ids, attn_mask, targets, definition_hidden_mean, sorted_indices  = _padding(batch, pad_token_id=tokenizer.pad_token_id, training=True)
                            current_train_dataset = TokenizationDataset(input_ids= tok_ids, targets = targets, attention_mask = attn_mask, definition_hidden_mean=definition_hidden_mean)
                            mini_train_batch_padded[maxlen] = batch_gen(current_train_dataset, batch_size, num_workers=2)
                            print('train_batch size and amount ', maxlen, len(batch))
                        else:
                            print('stage, current_maxlen and len of train batch', maxlen, len(batch) )

                    for l, valid_batch in mini_valid_batch_padded.items():
                        cur_model_dir = os.path.join(save_path, 'bert2span_{}_val_batch_{}'.format(r, l))
                        checkpoint_callback = callbacks.ModelCheckpoint(monitor='val_loss', dirpath=cur_model_dir, mode = 'min',
                                                    filename='bert2span-{epoch:02d}-{val_loss:.5f}', save_top_k=2)
                        trainer = Trainer(gpus=gpus, gradient_clip_val = 1.0, stochastic_weight_avg=True, max_epochs=max_epochs,callbacks=checkpoint_callback, precision=16)      
                        for maxlen, train_batch in mini_train_batch_padded.items():
                            #valid_batch = mini_valid_batch_padded[l]
                            trainer.fit(bert2span, train_batch, valid_batch)
                                #print('Best model path of task {}'.format(task + round), checkpoint_callback.best_model_path)
                        torch.cuda.empty_cache()


if __name__ == '__main__':
    import sys
    #config_path = 'configs/transformer_config.yaml'
    #main(config_path)
    assert len(sys.argv) >= 2, "The path to the config file must be given as argument!"
    main(sys.argv[1])

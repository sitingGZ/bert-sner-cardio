
import json
from typing import List, Dict
import pandas as pd
from collections import Counter
 
from copy import deepcopy
import torch
from transformers import  AutoTokenizer, AutoModelForMaskedLM
from modules.helpers import load_config

from modules.BERT2span_semantic_disam import BERT2span
from modules.helpers import CustomThread, get_host_name
from modules.inference import final_label_results_rescaled
from modules.helpers import load_config, set_seed

# UMLS term types
def getTerms(sem_df:pd.DataFrame, to_exclude: list =  ['T071', 'T051']) -> Dict:
    tuis, stys, defs  = sem_df['tui'], sem_df['sty'], sem_df['def']
    tuis_stys_defs_dict = {}
    tuis_stys_defs_dict = {t: (s,d) for t, s, d in zip(tuis, stys, defs) if t not in to_exclude} 
    return tuis_stys_defs_dict

def getTermLevel(level_df: pd.DataFrame) -> Dict:
    term_of_levels = level_df.groupby('Level')
    level_term_dict = {n[0]:n[1].values.tolist() for n in term_of_levels['TypeID']}
    return level_term_dict


def getTermGroup(sem_df:pd.DataFrame, level_df:pd.DataFrame) -> Dict:
    tui_levels = {t:l for t,l in zip(level_df['TypeID'], level_df['Level'])}
    term_of_group = sem_df.groupby('group')
    group_term_dict = {n[0]:[(t, tui_levels[t]) for t in n[1].values.tolist()] for n in term_of_group['tui']}
    return group_term_dict

# Cardiology tsv files reader
def retrieve_sentences(tsv_lines):
    sentences = {}
    for l in tsv_lines:
        if '\t' in l:
            l_split = l.split('\t')
            sent_idx = int(l_split[0].split('-')[0])
            if sent_idx not in sentences:
                sentences[sent_idx] = []
            sentences[sent_idx].append(l_split)
    return sentences

def get_labels_by_index(sents, idx):
    labels = []
    for sent in sents:
        print(sent)
        labels.append(sent[idx])
    return Counter(labels)

def get_annos_by_section(sentences):
    sections = {}
    for k, sents in sentences.items():
        sec = sents[0][-1]
        if sec not in sections:
            sections[sec] = []
        sections[sec].extend(sents)
    return sections

def token_to_sentences(section_sents):
    sentences = {}
    for sent in section_sents:
        sent_id = sent[0].split('-')[0]
        token = sent[2]
        if sent_id not in sentences:
            sentences[sent_id] = []
        sentences[sent_id].append(token)
    return sentences

def return_sections(tsv_file_path):
    tsv_lines = [l.strip() for l in open(tsv_file_path)]
    #lines[:10]
    sentences = retrieve_sentences(tsv_lines)
    sections = get_annos_by_section(sentences)
    valid_sections = [k for k in sections.keys() if len(k) > 1]
    return valid_sections, sections

def return_section_sentences(sec, sections):
    sec_sents = token_to_sentences(sections[sec])
    return sec_sents.values()

def prepare_model(config_file):
    bert_language = 'ger_bert'
    add_kldiv = False
    freeze = False # For german language model, updating the parameters achieve better performance
    configs = load_config(config_file)
    set_seed(seed=42)
    language = configs['data']['language']
    #base_name = base_names[bert_language]
    base_names = {'pubmed_bert': "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", 'ger_bert': "bert-base-german-cased"}
    base_name = base_names[bert_language]
    tokenizer_path = configs['model']['tokenizer'][language]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    bertMLM = AutoModelForMaskedLM.from_pretrained(base_name)
    
    bert2span = BERT2span(configs, bertMLM, tokenizer, freeze=freeze, add_kldiv=add_kldiv)
    #state_dict_path = "checkpoints/german_bert_ex4cds_500_semantic_term.ckpt"
    state_dict_path = configs['train']['checkpoint_path']
    state_dict = torch.load(state_dict_path, map_location=torch.device('cpu'))
    bert2span.load_state_dict(state_dict)
    return bert2span, tokenizer

# Predict the ner results and scores for constructing heatmap for semantic labels
def get_prediction(tokenizer, model, word_list, semantic_labels, threshold):
        if type(semantic_labels) == list:
            semantic_labels = {l:l for l in semantic_labels}
        heatmap_scores_dict, ner_results = final_label_results_rescaled(word_list, tokenizer, model, suggested_terms=semantic_labels, threshold=threshold)
        return heatmap_scores_dict, ner_results
    
   
# Retrieve biomedical ontologies




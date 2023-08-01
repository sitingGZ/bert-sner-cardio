from copy import deepcopy
import torch


from transformers import AutoConfig, AutoTokenizer, AutoModelForMaskedLM, EncoderDecoderConfig
from modules.BERT2span_semantic_disam import BERT2span
from modules.helpers import load_config, set_seed, html_heatmap


suggested_terms = {'Condition': 'Zeichen oder Symptom',
 'DiagLab': 'Diagnostisch und Laborverfahren',
 'LabValues': 'Klinisches Attribut',
 'HealthState': 'Gesunder Zustand',
 'Measure': 'Quantitatives Konzept',
 'Medication': 'Pharmakologische Substanz',
 'Process': 'Physiologische Funktion',
 'TimeInfo': 'Zeitliches Konzept'}

def inference_valid_indices(words, tokenizer, label_token_tags_dict):
    #words = seq_label_pair[0]
    #tags = seq_label_pair[1]
    #current_labels = [tag for tag in tags if tag != 'O']
    #targets = {}
    #encoded_tokens = []
    valid_indices = []
    
    current_token_tags_dict = deepcopy(label_token_tags_dict)
    for i, w in enumerate(words):
        encoded = tokenizer.encode(w)[1:-1]
        #encoded_tokens.extend(encoded)
        valid = [0] * len(encoded)
        valid[0] = 1
        valid_indices.extend(valid)
            
        for tag, (input_tokens, current_tags) in current_token_tags_dict.items():
            input_tokens += encoded
           
    return current_token_tags_dict,  valid_indices


def resolve_prediction_inference(scores_dict, label_tag, label_max_score_dict, threshold= 0.5):
    label_keys = list(scores_dict.keys())
    #puncts_idx = [d for d, w in enumerate(words) if w not in [',', '.', ';', '!', '?']]
    length = len(scores_dict[label_keys[0]])
    predict_label_resolved = [0] * length
    #for i, v in enumerate(valid_indices):
        #if v > 0:
    for i in range(length):
            #preds = [(scores_dict[k][0] + scores_dict[k][1][i])/3 for k in scores_dict]
            preds = [scores_dict[k][i]/label_max_score_dict[k] for k in scores_dict]
            #print(preds)
            max_s = max(preds)
            if max_s > threshold:
                max_idx = preds.index(max_s)
                tag = label_tag[label_keys[max_idx]]
                predict_label_resolved[i] = tag
    #predict = [p for i, p in enumerate(predict_label_resolved) if valid_indices[i] > 0]
    return predict_label_resolved

def get_prediction_heatmap_and_score(label, words, valid_indices, input_ids, attention_mask, bert2span_model,tokenizer):
    prediction = bert2span_model(input_ids = input_ids, attention_mask = attention_mask)
   
    results = [prediction.detach().cpu().numpy()[0][t][0] for t in range(input_ids.shape[1])]
    assert len(results) > len(valid_indices), 'Results are less than input tokens, why, label, input ids length {} {} {}, valid indices {}'.format(label, input_ids.shape[1], len(results), len(valid_indices))
    length_label_scores = len(results) - len(valid_indices)
    label_scores = results[:length_label_scores]
    #print(label_scores)
    heatmap_scores =label_scores[:2] + [label_scores[-1]] + [s for i, s in enumerate(results[-len(valid_indices):]) if valid_indices[i] > 0]
    valid_results = [label_scores[1] * s for i, s in enumerate(results[-len(valid_indices):]) if valid_indices[i] > 0 ]
    
    spans = [(t, r, tokenizer.decode(input_ids[0][t])) for t, r in enumerate(results)]
    
    heatmap_words = ['[CLS]', label, '[SEP]'] + words
    #scores = [r  for t, r in enumerate(results)]
    #print(spans)
    #heatmaps,scaled_scores = html_heatmap(words, scores)
    return heatmap_words,heatmap_scores, spans, valid_results

def label_token_tags(label_words, tokenizer):
    targets = {}
    for label_word in label_words:

        input_tokens = tokenizer.encode(label_word)

        current_tags = [0]
        current_tags += [1]* (len(input_tokens)-2)
        current_tags += [0]
        targets[label_word] = (input_tokens, current_tags)
    return targets 

def final_label_results_rescaled(words_list, tokenizer, bert2span, suggested_terms, threshold=0.5):
    
    #heatmaps_list = []
    label_results = {label: [] for t, label in suggested_terms.items()}
    heatmap_pairs = {label:[] for t, label in suggested_terms.items()}
    label_max_pred_score = {label: 0 for t, label in suggested_terms.items()}
    label_token_tags_dict = label_token_tags(suggested_terms.values(), tokenizer)
    
    # Predict on words of one sentence from all tokenized sentences
    for words in words_list:
        label_input_tokens, valid_indices = inference_valid_indices(words=words, tokenizer=tokenizer, label_token_tags_dict=label_token_tags_dict)
        for label, (input_ids, _) in label_input_tokens.items():           
            attention_mask = [1] * len(input_ids)
            attention_mask = torch.Tensor(attention_mask).unsqueeze(0)
            input_ids = torch.LongTensor(input_ids).unsqueeze(0)                    
            heatmap_words, heatmap_scores, spans, results = get_prediction_heatmap_and_score(label, words, valid_indices, input_ids, attention_mask, bert2span,tokenizer)
            if max(results) > label_max_pred_score[label]:
                label_max_pred_score[label] = max(results)
            label_results[label].append(results)
            heatmap_pairs[label].append((heatmap_words, heatmap_scores))
            
    ner_label_tags = {'O': 0}
    ner_label_tags.update({l : i+1 for i, l in enumerate (list(suggested_terms.values()))})
    
    suggested_terms_tags = {l:t for t,l in suggested_terms.items()}
    
    ner_tag_labels = {i:suggested_terms_tags[l] if l in suggested_terms_tags else l for  l,  i in ner_label_tags.items() }

    ner_results = []
    
    for s in range(len(words_list)):
        
        score_dict = {k: label_results[k][s] for k in label_results.keys()}    
        predict_label_resolved = resolve_prediction_inference(score_dict, ner_label_tags,  label_max_score_dict= label_max_pred_score, threshold=threshold)
        current_predict_result = [(w, predict_label_resolved[i], ner_tag_labels[predict_label_resolved[i]]) for i, w in enumerate(words_list[s])]
        ner_results.append(current_predict_result)
        
    return heatmap_pairs, ner_results
            

def main(config_file, sentence_file, suggested_terms = suggested_terms, threshold=0.5):
    #config_file = "configs/step3_gpu_span_semantic_group.yaml"
    bert_language = 'ger_bert'
    add_kldiv = False
    freeze = False # For german language model, updating the parameters achieve better performance
    configs = load_config(config_file)
    set_seed(seed=42)
    dataset = configs['data']['dataset']
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
    
    words_list = [l.strip().split() for l in open(sentence_file)]
    heatmap_list, ner_results = final_label_results_rescaled(words_list, tokenizer, bert2span, suggested_terms=suggested_terms, threshold=threshold)
    print(heatmap_list)
    print(ner_results)
    
    
    
    

if __name__ == '__main__':
    import sys
    #config_path = 'configs/transformer_config.yaml'
    #main(config_path)
    assert len(sys.argv) >= 2, "The path to the config file must be given as argument!"
    main(sys.argv[1], sys.argv[2])
    
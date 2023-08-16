
from dash import html, dcc, Dash
import dash_bootstrap_components as dbc
from dash import Output, Input, State
import dash_mantine_components as dmc

import os, json
# Dataset utils, 
# Predicted results

def get_result_cardio_files(result_path):
    prediction_file_paths = {f.split('_')[0]: f for f in os.listdir(result_path) if '_score' in f}
    classification_file_paths = {f.split('_')[0]: f for f in os.listdir(result_path) if 'ner_' in f}
    return prediction_file_paths, classification_file_paths

def return_json_cardio_dict(tsv_id,result_path, prediction_file_paths, classification_file_paths):
    if type(tsv_id) != 'str' or 'tsv' not in tsv_id:
        tsv_id = '{}.tsv'.format(tsv_id)
    prediction_dict = json.load(open(os.path.join(result_path, prediction_file_paths[tsv_id])))
    classification_dict = json.load(open(os.path.join(result_path, classification_file_paths[tsv_id])))
    return prediction_dict, classification_dict

# Ex4CDS token spans 
def get_ex4cds_text_id(json_file_name):
    current_file = json_file_name.split('.')[0]
    sec_name = current_file.split('_')[0]
    text_id = '_'.join(current_file.split('_')[1:-2])
    return sec_name, text_id

def get_ex4cds_prediction(result_main_path, current_json_file):
    classification_0 = json.load(open(os.path.join(result_main_path, current_json_file)))
    tokens = [[w[0] for w in l ] for l in classification_0['Physical Object']['0.5']]
    return tokens, classification_0

def get_ex4cds_text_lines(ex4cds_path, section, text_id):
    #print(ex4cds_path, section, text_id+'.txt')
    lines = [l.strip().split() for l in open(os.path.join(ex4cds_path, section, text_id+'.txt'))]
    #print(lines)
    word_spans = []
    for ws in lines:
        current_spans = {}
        start = 0
        end = 0
        for i, w in enumerate(ws):
            if i < len(ws)-1:
                end = start + len(w)
                current_spans[start] = (end, w)
                #current_spans.append((w, start, end))
                start = end+1
            else:
                end = start + len(w)
                current_spans[start] = (end, w)
                #current_spans.append((w, start, end))
            
            #print(i, w, end, start)
        word_spans.append(current_spans)
    #print('word spans in function  get_ex4cds text lines, ', word_spans)
    return word_spans, lines

def map_token_spans(original_text_spans, current_tokens):
    
    token_spans = []
    #spans = {w[1]: (w[2], w) for i, w in enumerate(original_text_spans)}
    #print(spans)
    
    #print('original text spans ', original_text_spans)
    original_text_spans = {int(k):v for k, v in original_text_spans.items()}
    #print('original text spans int (k)', original_text_spans)
    start = 0
    span_end = original_text_spans[start][0]
    original_text = original_text_spans[start][1]
    t = 0
    while t < len(current_tokens):
        current_t = current_tokens[t]
        if current_t not in original_text:
            #print(current_t)
            
            #while t < len(tokens) -1 :
            if t < len(current_tokens) -1:
                next_t = current_tokens[t+1]
                if next_t in original_text:
                    #print(original_text_spans)
                    idx_next = original_text.index(next_t)
                    #print(original_text, t_end, start, idx_next)
                    current_t = original_text[start-t_end-1:idx_next]
                    
                
                else: 
                    current_t = original_text[start-t_end-1:]
                    #print('the current t for unknow last to the end of the original text', current_t)
                    #print(original_text, t_end, start,)
                
                    #break
            else:
                current_t = original_text[start-t_end-1:]
                
            #print(current_t, next_t)
            #print('the current t for unknow ', current_t)
            
            
           
        t_end = start + len(current_t)
        if t_end < span_end:
                token_spans.append((start, t_end, current_t))
                start = t_end
                t += 1
                #print(token_spans, start, t_end, t)
        elif t_end == span_end:
                token_spans.append((start, t_end,current_t))
                start = t_end+1
                t += 1
                #print(token_spans, start,t_end, t)
                if start in original_text_spans:
                    span_end = original_text_spans[start][0]
                    original_text = original_text_spans[start][1]
                    
                
                    
        else:
                print('\n attention !!!!!!', t_end, span_end, start, t)
                #print(current_tokens,original_text_spans)
                #print(token_spans)
                t += 1
        
            #assert t_end == span_end, "last end {} should be equal to span end {} of start {} at token {}".format(t_end, span_end, start, t)
    return token_spans
        
        


def bert_tokenize(text_line, bert_tokenizer):
    tokenized = bert_tokenizer.encode(text_line)[1:-1]
    decoded = [bert_tokenizer.decode(t) for t in tokenized]
    
    rebuilt = []
    current_t = decoded[0:1]
    for i, t in enumerate(decoded):
        if i > 0:
            if '##' in t:
                current_t.append(t[2:])
            else:
                rebuilt.append(''.join(current_t))
                current_t = decoded[i:i+1]
    rebuilt.append(''.join(current_t))
    return rebuilt

def get_spans_all_sentences_ex4cds(lines, word_spans, tokens):
    spans_all_sentences = {}
    print('word spans in get spans all sentece ex4cds ', word_spans)
    for i in range(len(lines)):
        
        sent_spans = map_token_spans(word_spans[i], tokens[i])
        spans_all_sentences[i] =  {'text': ' '.join(lines[i]), 'text_spans': sent_spans, 'ner_spans': []}
    return spans_all_sentences
   
#dash_html_heatmap(words, scores,cmap_name="bwr")
def get_prediction_classification_section(prediction_scores, classification_dict,sec_tab):
    scores_dict_sec = prediction_scores[sec_tab]
    classification_dict_sec = classification_dict[sec_tab]
    
    sentence_ids = [i for i in scores_dict_sec['sentence_ids']]
    return sentence_ids, scores_dict_sec, classification_dict_sec

# Make original text and spans from cardiode tsv data 
def make_orginal_text_spans(sents):
    """
    [['519-1', '24327-24332', 'Ärztl', '_', '_', '_', 'Abschluss'],
      ['519-2', '24332-24333', '.', '_', '_', '_', 'Abschluss'],
      ['519-3', '24334-24342', 'Direktor', '_', '_', '_', 'Abschluss'],
      ['519-4', '24343-24351', 'Oberarzt', '_', '_', '_', 'Abschluss'],
      ['519-5', '24352-24364', 'Stationsarzt', '_', '_', '_', 'Abschluss']]
    """
    text_str = ''
    spans = [None] * len(sents)
    start = 0

    for i, s in enumerate(sents):
        if i == 0:
            text_str += s[2]
    
            end = len(s[2])
            spans[i] = (start, end)
        else:
            previous_end = sents[i-1][1].split('-')[1]
            current_start = s[1].split('-')[0]
            text_str += ' ' * (int(current_start) - int(previous_end))
            start = len(text_str)
            text_str += s[2]
            end = start + len(s[2])
            spans[i] = (start, end)
        
    return text_str, spans
 

def get_spans_all_sentences(sentence_ids, tsv_sentences):
    spans_all_sentences = {}
    for i, sent_id in enumerate(sentence_ids):
        text_str, sent_spans = make_orginal_text_spans(tsv_sentences[sent_id])
        spans_all_sentences[i] =  {'text': text_str, 'text_spans': sent_spans, 'ner_spans': []}
    return spans_all_sentences

def get_sentence_classification(classification_dict_sec, spans_all_sentences, selected_sem_types, threshold='revision'):
    """
    """
    
    semantic_levels = {k : i for i, k in enumerate(list(classification_dict_sec.keys()))}
    for k, level in semantic_levels.items():
        if threshold in classification_dict_sec[k]:
            classification = classification_dict_sec[k][threshold]
        else:
            classification = classification_dict_sec[k]['0.5']
        #print(k)
        for s, sent in enumerate(classification):
            #spans_all_sentences[s]['ner_spans'] = []
            sent_spans = spans_all_sentences[s]['text_spans']
            #print(len(sent_spans), len(sent))
            
            current_ents = {}
            for t, w in enumerate(sent):
                if w[2] in selected_sem_types:
                    if w[2] not in current_ents:
                        current_ents[w[2]] = []
                    current_ents[w[2]].append(t)
            
            #print(current_ents) 
            if len(current_ents) > 0:
                for ent, ts in current_ents.items():
                    start = sent_spans[ts[0]][0]
                    end = sent_spans[ts[0]][1]
                    #print(ent, ts, start, end)
                    if len(ts) > 1:
                        for d in range(1, len(ts)):
                            if ts[d] - ts[d-1] == 1:
                                #print(d)
                                end = sent_spans[ts[d]][1]
                                if d == len(ts)-1:
                                    #print('d == len(ts)-1', ent, start, end )
                                    #if end < start:
                                        #print('in the 1, len(ts) range ' ,s, ent, ts)
                                    spans_all_sentences[s]['ner_spans'].append((start, end, ent))
                                    #print(spans_all_sentences[s])
                            else:
                                #end = sent_spans[ts[d-1]][1]
                                #print('new ent', ent, start, end )
                                #if end < start:
                                        #print('out of the 1, len(ts) range ' ,s, ent, ts)
                                spans_all_sentences[s]['ner_spans'].append((start, end, ent))
                                start = sent_spans[ts[d]][0]
                                end = sent_spans[ts[d]][1]
                    else:
                        #if end < start:
                                        #print('len (ts )== 1 ' ,s, ent, ts)
                        spans_all_sentences[s]['ner_spans'].append((start, end, ent))
                
                        
    return spans_all_sentences

 

 
# App utils
from dashpages.extraction import getTermGroup

colors = [
    "coral",
    "navy",
    "brown",
    "red",
    "pink",
    "grape",
    "violet",
    "indigo",
    "blue",
    "lime"
    "icebert",
    "celeste",
    "jade",
    "maroon",
    "emerald",
    "olive",
    "cream",
    "blonde",
    "peach",
    "gold",
    "beer",
    "sage",
    "brass",
    "burgundy",
    "rose",
    "bronze",
    "wood",
    "taupe",
    "coffee",
    "sepia",
    "mahogany",
    "sedona",
    "blush"]
# Init navbar


# Column left Sidebar
def make_switch(label, color,i, value=True):
    return dbc.Switch(label=label, label_style={'color':color}, value=value, id="{}_{}".format(i, label))

added_badge_group = dmc.Stack(children = [],
id = "added_badge_group")

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": "13rem",
    "left": 0,
    "bottom": 0,
    "margin-left": "1rem",
    "width": "50rem",
    "padding": "1rem",
    "overflow-y":'auto',
    "background-color": "#f8f9fa"}

"""
add_more_types = dmc.Button("Add selected types", id="open-button")

def make_add_more_types_multi(group_tuis, types_en, types_de):
    html_div_children = []
    html_div_dropdown_ids = {}
    for group, tuis in group_tuis.items():
        label = dbc.Label(group, html_for="{}-dropdown".format(group))
        group_dropdown = dcc.Dropdown(
                id="{}-dropdown".format(group),
                options=[{"value": types_de[tui][0], "label": '{} ({})'.format(types_de[tui][0],types_en[tui][0]) } for tui in tuis if tui in types_en and tui in types_de],
                value= [],
                multi=True,
            )
        html_div_dropdown_ids[group] = "{}-dropdown".format(group)
        html_div_children.extend([label, group_dropdown])
    return html.Div(id='add-types-dropdowns', children=html_div_children), html_div_dropdown_ids


def make_sidebar(suggested_terms, section_checklist, group_tuis, types_en, types_de):
    add_more_types_dropdowns, dropdow_ids = make_add_more_types_multi(group_tuis, types_en=types_en, types_de=types_de)
    sidebar_children = [dbc.Row([ dbc.Col([html.H5("Select sections", ), 
                                dbc.Checklist(options = [{'label':sec, 'value':sec } for sec in section_checklist], id='section-checklist', value=[])
                                ]),  
                       dbc.Col([html.H5("Suggested semantic types", ), 
                    dmc.Stack([make_switch(label, suggested_colors[i],i) for i, label in enumerate(list(suggested_terms.values()))],
    id = "label-badge-group")])]),
              html.Hr(), 
              html.H5("Add more semantic types by groups", ),
              add_more_types_dropdowns, html.Br(), 
              add_more_types]
    return sidebar_children, dropdow_ids
    
    
SECTIONS = ['Anrede', 'AktuellDiagnosen', 'Diagnosen', 'AllergienUnverträglichkeitenRisiken', 'Anamnese', 'AufnahmeMedikation', 'KUBefunde', 'Labor', 'Befunde', 'EchoBefunde', 'Zusammenfassung', 'EntlassMedikation', 'Abschluss']
# Column middle Main content
CONTENT_STYLE = {
    "bottom": 0,
    "margin-left": "52rem",
    "margin-right": "1rem",
    "padding": "1rem 1rem",
    'borderWidth': '1px',
    'borderStyle': 'double',
    'borderRadius': '5px',
    }

upload_file = dcc.Upload([
        'Drag and Drop or ',
        html.A('Select a tsv file from')
    ], style={
        'width': '100%',
        'height': '30px',
        'lineHeight': '30px',
        'borderWidth': '1px',
        'borderStyle': 'dashed',
        'borderRadius': '5px',
        'textAlign': 'center'
    })

def make_one_section_tab(sec_title, sentences):
    #show_bt = dbc.Button('Show Entity Recognition Results', color='success', id='{}-show-bt'.format(sec_title))
    #sentences = [' '.join(sent) for sent in sentences]
    #textarea = dmc.Textarea( value = '\n'.join(sentences),  minRows = 20, )
    textarea = html.Div(children= [html.P(s) for s in sentences], style = {'borderWidth': '1px',
        'borderStyle': 'dotted',
        'borderRadius': '5px',"padding": "1rem 1rem", "background-color": "#ffffff"})
    return dbc.Tab(textarea, label = sec_title, activeLabelClassName="text-success", label_style={})

def make_one_explain_table(sec_title, sentences):
    #show_bt = dbc.Button('Show Entity Recognition Results', color='success', id='{}-show-bt'.format(sec_title))
    #sentences = [' '.join(sent) for sent in sentences]
    #textarea = dmc.Textarea( value = '\n'.join(sentences),  minRows = 20, )
    textarea = html.Div(children= [html.Span(s) for s in sentences], style = {'borderWidth': '1px',
        'borderStyle': 'dotted',
        'borderRadius': '5px',"padding": "1rem 1rem", "background-color": "#ffffff"})
    return dbc.Tab(textarea, label = sec_title)
     

# Upload and make sections 
def make_section_tabs(section_dict, show_bt):
    tabs = []
    tab_show_bt_ids = {}
    for sec, sec_sentences in section_dict.items():
        sec_card, sec_title, show_bt_id = make_one_section_card(sec, sec_sentences['text'], show_bt)
        tabs.append(dbc.Tab(sec_card, label = sec_title))
        tab_show_bt_ids[sec] = show_bt_id
    section_tabs = dbc.Tabs(tabs)
    return section_tabs, tab_show_bt_ids

def make_section_extraction_results(section_dict):
    tabs = []
    tab_show_bt_ids = {}
    for sec, sec_sentences in section_dict.items():
        tokens = sec_sentences['token']
        #sec_card, sec_title, show_bt_id = make_one_section_card(sec, sec_sentences)
        
        tabs.append(dbc.Tab(sec_card, label = sec_title))
        #tab_show_bt_ids[sec] = show_bt_id
    section_tabs = dbc.Tabs(tabs)
    return section_tabs, tab_show_bt_ids

"""
    
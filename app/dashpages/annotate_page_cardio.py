import copy
import dash
from dash import dash_table

from dash import no_update
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
from collections import OrderedDict

from dash import dcc, html, callback, clientside_callback, ctx

from dash import MATCH, ALL
from dash.dependencies import Output, Input, State
from dash_extensions import Purify
import ast
import numpy as np

from flask import Flask
import pathlib
import pandas as pd
import json 
import os
import logging
import matplotlib.pyplot as plt
import itertools

from dashpages.utils import make_switch, get_spans_all_sentences
from dashpages.utils import get_ex4cds_text_id, get_ex4cds_prediction, get_ex4cds_text_lines, map_token_spans, get_spans_all_sentences_ex4cds
from dashpages.format_markup import dash_html_prediction_score
from dashpages.span_markup import format_span_line_markup
from dashpages.metrices import get_amount_sentences, get_amount_annotations,  get_section_sentences_dict
from dashpages.metrices import get_y_pred_true, get_group_classification_report, make_report_table

from dashpages.format_markup import Palette, colors_ent, color_fact, color_temporal,color_risk_att

from copy import deepcopy
import torch
from transformers import  AutoTokenizer, AutoModelForMaskedLM
import plotly.express as px


ANNOTATEPATH = {'span': 'Kardiologie', 'href':'/annotate_cardio_success'}
SIGGINPATH = {'span': 'Kardiologie', 'href':'/annotate_cardio_login'}
# Preparing the model to interact with the user interface
BASE_PATH = pathlib.Path(__file__).parent.resolve()
#DATA_PATH = BASE_PATH.joinpath("cardiode").resolve()
RESULT_PATH_cardio = BASE_PATH.joinpath("cardiode/finetuned_ex4cds/shot_500").resolve()
classification_csv = os.path.join(RESULT_PATH_cardio, "classification.csv")
user_csv_path = os.path.join(RESULT_PATH_cardio, 'users.csv')
definition_file_german = os.path.join(BASE_PATH, "data/tui_sem_dict_2001_german_removed_umlaut.json")
#CLASSIFICATION_DB = pd.read_csv(classification_csv)
#USER_DB = pd.read_csv(user_csv_path)

USER_DB = json.load(open(os.path.join(RESULT_PATH_cardio, 'users.json')))

#ids_section_path = os.path.join(BASE_PATH, "to_annotate_ids.xlsx")
#ids_df = pd.read_excel(ids_section_path)

IDEAL_SECTIONS = ['Befunde', 'Diagnosen', 'Zusammenfassung']
IDEAL_SECTIONS_IDS = {'Befunde': ['118', '122', '174', '190', '197', '198', '218', '225', '235', '247', '263', '275', '288'], 
                      'Diagnosen': ['1', '13', '106', '109', '119', '132', '133', '150', '164', '165', '172'], 
                      'Zusammenfassung': ['21', '101', '135', '149', '151', '209', '210', '219', '249', '251', '256', '259', '262', '264', '296', '311', '322']}

#all_sections = list(set(CLASSIFICATION_DB['Section']))

#annotators = ['BERT-SNER', 'User1']

TSV_PATH = BASE_PATH.joinpath('cardiode/tsv/CARDIODE400_main').resolve()
dfki_colors = ["#2980b9", "#3498db"]

#THRESHOLDS = {'0.5':'threshold_05', '0.6':'threshold_06', '0.7':'threshold_07'}
THRESHOLD= '0.5'

ANNOTATEPATH = {'span': 'Kardiologie', 'href':'/annotate_cardio'}

multi_layer_suggested_terms = {'Physical Object': 
                               {'T017': 'Anatomie',
                                "T200": "Arzneimittel", 
                                "T103": "Chemikalien",
                             "T074": "Medizinisches Geraet"
                               }, 
                               " Conceptual Entity": {
                                   'T201': "Klinisches Attribut",
                                   'T081': "Quantitatives Konzept",
                                   'T184': "Zeichen oder Symptom", 
                                   'T034': "Labor- oder Testergebnis", 
                                   'T079': "Zeitliches Konzept"},
                               "Activity" : {
                                  'T059': "Laborverfahren", 
                                   'T060': "Diagnostisches Verfahren",  
                                   'T061':"Therapeutisches oder praeventives Verfahren"}, 
                               "Phenomenon or Process":{
                                'T037': "Verletzung oder Vergiftung",
                               'T039': "Physiologische Funktion", 
                               'T046': "Pathologische Funktion", 
                               'T047': "Krankheit", 
                               'T048': "Psychische oder Verhaltensstoerung"},
                               "Health State":{'T300': "Gesunder Zustand" ,  
                                               "T301": "Verschlechter Zustand"},
                               "Factuality": {'T400': 'kein oder negiert', 'T401' : 'gering',  'T402' : 'fraglich','T403': 'zukueftig', 'T404': 'unwahrscheinlich'},
                               "Temporal": {'T500': 'aktuelles Ereignis', 'T501': 'Vergangenheit zum aktuellen Ereignis','T502': 'vergangenes Ereignis','T503': 'zukuenftiges Ereignis'}}

TERM_DEF_DB = json.load(open(definition_file_german))

FACTUALITY = {'T400': 'kein oder negiert', 'T401' : 'gering',  'T402' : 'fraglich','T403': 'zukueftig', 'T404': 'unwahrscheinlich'}

TEMPORAL = {'T500': 'aktuelles Ereignis', 'T501': 'Vergangenheit zum aktuellen Ereignis','T502': 'vergangenes Ereignis','T503': 'zukuenftiges Ereignis'}

RISK_FACTORS = {'increase':'Risiko erhöhen', 'decrease': 'Risiko verringern', 'not relevant': 'kein Risiko'}

#SUGGESTED_TYPES_ENT = list(Ex4CDS_TYPES_ENTITY.values())
#SUGGESTED_TYPES_ATT = list(Ex4CDS_TYPES_KONZEPT.values())
SECTION_LIST = ['Anrede', 'AktuellDiagnosen', 'Diagnosen', 'AllergienUnverträglichkeitenRisiken', 'Anamnese', 'AufnahmeMedikation', 'KUBefunde', 'Labor', 'Befunde', 'EchoBefunde', 'Zusammenfassung', 'EntlassMedikation', 'Abschluss']
SECTIONS_NEPH =  ['Rejection', 'Transplantatversagen', 'Infection']

def get_tsv_lines(tsv_id):
    if type(tsv_id) != 'str' or 'tsv' not in tsv_id:
        tsv_id='{}.tsv'.format(tsv_id)
    tsv_lines = [l.strip() for l in open(TSV_PATH.joinpath(tsv_id))]
    return tsv_lines

def retrieve_sentences(tsv_lines):
    sentences_dict = {}
    for l in tsv_lines:
        if '\t' in l:
            l_split = l.split('\t')
            sent_idx = l_split[0].split('-')[0]
            if sent_idx not in sentences_dict:
                sentences_dict[sent_idx] = []
            sentences_dict[sent_idx].append(l_split)
    return sentences_dict

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

def get_scores_dict_by_sec(scores_dict, sec_tab):
    scores_dict_sec = deepcopy(scores_dict[sec_tab])
    #print(scores_dict_sec)
    sentence_ids = scores_dict_sec['sentence_ids']
    return scores_dict_sec, sentence_ids

def get_sentence_scores_by_type_cardio(scores_dict_sec, sentence_ids, tsv_sentences, sem_types, ca = None):
    """
    :param sem_types: selected types in all category [type1, type2])
    
    """
    #category, types = sem_types[0], sem_types[1]
    scores_types = {}
    
    for ty in sem_types:
        if ca is not None and ca in scores_dict_sec:
            current_scores = scores_dict_sec[ca]
            if ty in current_scores: 
                    scores = current_scores[ty]
                    ty_words = ty.split()
                    scores_types[ty] = []
                    for i, d in enumerate(sentence_ids):
                        current_sent = tsv_sentences[d]
                        words = [l[2] for l in current_sent]
                
                        ty_scores = [float(s) for s in scores[i][:len(ty_words)]]
                        word_scores = [float(s) for s in scores[i][len(ty_words):]]
                        #print(len(ty_scores), len(word_scores), len(words), len(scores[i]))
                        scores_types[ty].append({'type_scores': (ty_words, ty_scores), 'word_scores': (words, word_scores)})
        else:
            for ca, current_scores in scores_dict_sec.items():
                if type(current_scores) == dict:
                    #print(current_scores)
                    if ty in current_scores: 
                        scores = current_scores[ty]
                        ty_words = ty.split()
                        scores_types[ty] = []
                        for i, d in enumerate(sentence_ids):
                            current_sent = tsv_sentences[d]
                            words = [l[2] for l in current_sent]
                
                            ty_scores = [float(s) for s in scores[i][:len(ty_words)]]
                            word_scores = [float(s) for s in scores[i][len(ty_words):]]
                        #print(len(ty_scores), len(word_scores), len(words), len(scores[i]))
                        scores_types[ty].append({'type_scores': (ty_words, ty_scores), 'word_scores': (words, word_scores)})
    return scores_types

def make_section_options(sections_useful, all_sections = None):
    # CLINIC_DOMAIN = {'Cardiology':'Kardilogie', 'Nephrology':'Nephrologie'}
    
        #assert sections_useful is not None, "Sections for domain {} is required.".format(sections_useful)
        #return list(prediction_json_dict.keys())
        if type(sections_useful) != list:
            sections_useful = list(sections_useful)
        options = []
        if all_sections is not None:
            for s in all_sections:
                if s not in sections_useful:
                    options.append({'label': s, 'value': s, 'disabled': True})
                else:
                    options.append({'label': s, 'value' :s})
        else:
            options = [{'label':s, 'value':s} for s in sections_useful]
        return options, sections_useful
    
def get_prediction_section(prediction_scores, section):
    scores_dict_sec = prediction_scores[section]
    sentence_ids = [i for i in scores_dict_sec['sentence_ids']]
    return sentence_ids, scores_dict_sec

    
def get_section_sentences(sec_sentences_ids, tsv_sentences_dict):
    #ids = [d for d in sec_sentences_ids]
    sentences = []
    num_of_sentences = len(sec_sentences_ids)
    num_of_words = 0
    for d in sec_sentences_ids:
        sentences.append(' '.join([l[2] for l in tsv_sentences_dict[d]]))
        num_of_words += len(tsv_sentences_dict[d])
    return sentences, num_of_sentences, num_of_words

def get_classification_annotator(classification_sec_db, annotator="", group=""):
    """
    """
    #semantic_levels = {k : i for i, k in enumerate(list(classification_dict_sec.keys()))}
    if annotator not in classification_sec_db['Annotator']:
        annotator = "BERT-SNER"
    
    classification_group = classification_sec_db[classification_sec_db['Group'] == group]
    classification_annotator = classification_group[classification_group['Annotator']==annotator]
    return classification_annotator


def get_sentence_classification(classification_annotator):
    """
    """
    sentences_classification = {}
    for idx in set(classification_annotator['Sentence_idx']):
        sent_classification_db = classification_annotator[classification_annotator['Sentence_idx'] == idx]
        sent_classification = sent_classification_db.iloc[0]['Classification']
        if type(sent_classification) != list:
                sent_classification = ast.literal_eval(sent_classification)
        sentences_classification[int(idx)] = sent_classification
    return sentences_classification

def get_sentences_spans_ner(sentences_classification, spans_all_sentences, selected_sem_types):
    
    if type(sentences_classification) == list:
        sentences_classification = {idx: classification for idx, classification in enumerate(sentences_classification)}
           
        for idx, sent_classification in sentences_classification.items():
            sent_spans = spans_all_sentences[int(idx)]['text_spans']
            current_ents = {}
  
            for t, w in enumerate(sent_classification):
                #print(w)
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
                                    spans_all_sentences[int(idx)]['ner_spans'].append((start, end, ent))
                                    #print(spans_all_sentences[s])
                            else:
                                #end = sent_spans[ts[d-1]][1]
                                #print('new ent', ent, start, end )
                                #if end < start:
                                        #print('out of the 1, len(ts) range ' ,s, ent, ts)
                                spans_all_sentences[int(idx)]['ner_spans'].append((start, end, ent))
                                start = sent_spans[ts[d]][0]
                                end = sent_spans[ts[d]][1]
                    else:
                       
                        spans_all_sentences[int(idx)]['ner_spans'].append((start, end, ent))
        return None


def make_types_checkbox(group_colors, multi_layer_suggested_terms):
    semantic_types_checkbox_groups = {}
    for group, types in multi_layer_suggested_terms.items():
        semantic_types_checkbox_groups[group] =  dbc.Checklist(
            options=[{'label': t + ': ' + name, 'value':t} for t, name in types.items()
            ],
            label_style = {'color': group_colors.get(group).line.value} ,   
            value=list(types.keys()),
            inline=True,
            id="annotate-{}-group".format('-'.join(group.lower().split())),
            )
    return semantic_types_checkbox_groups

def make_select_types_to_explain(sec_scores_dict):
    categories = [list(value.keys()) for ca, value in sec_scores_dict.items() if type(value) == dict]
    categories_flat = list(itertools.chain.from_iterable(categories))
    return dcc.Dropdown(id='annotate-types-options', options = [{'label':ty, 'value':ty} for ty in categories_flat], multi=True)

def make_annotation_checklist(classification_dict, section, sem_group, threshold='0.5'):
    current_classification_dict = classification_dict[section][sem_group][threshold]
    sentence_checklist = {}
      
#text_id_sections_df = pd.DataFrame([{'id':f.split('.')[0], 'section':f.split('.')[2] } for f in os.listdir(DATA_PATH) ])
CLINIC_DOMAIN = {'Cardiology':'Kardiologie', 'Nephrology':'Nephrologie'}

select_domain = html.Div([
    html.H4('Domain'), 
    dcc.Dropdown([d for d in CLINIC_DOMAIN.values()], id='annotate-select-domain', multi=False)
    ], style= {'textAlign': 'left'})

# Third column
colors = {l: colors_ent[i] for i, l in enumerate(list(multi_layer_suggested_terms.keys()))}
colors_fact = {l:color_fact[0] for i, l in enumerate(list(FACTUALITY.keys()))}
colors_temp = {l:color_temporal[0] for i, l in enumerate(TEMPORAL.keys())}
colors.update(colors_fact)
colors.update(colors_temp)

PALETTE_ent = Palette([], colors)
colors_types = {}
for i, l in enumerate(list(multi_layer_suggested_terms.keys())):
    for ty in multi_layer_suggested_terms[l]:
        colors_types[ty] = PALETTE_ent.get(l)

PALETTE_types = Palette([], colors_types)
PALETTE_fact = Palette(color_fact)
PALETTE_temp = Palette(color_temporal)
PALETTE_risk_att = Palette(color_risk_att)

colors_suggested_ent = {e: PALETTE_ent.get(e).name for i, e in enumerate(list(multi_layer_suggested_terms.keys()))}
#colors_suggested_att = {e: PALETTE_att.colors[0].name for i, e in enumerate(SUGGESTED_TYPES_ATT)}
colors_suggested_fact = {e: PALETTE_fact.colors[0].name for i, e in enumerate(list(FACTUALITY.keys()))}
colors_suggested_temp = {e: PALETTE_temp.colors[0].name for i, e in enumerate(list(TEMPORAL.keys()))}
colors_suggested_risk = {e: PALETTE_risk_att.colors[i].name for  i, e in enumerate(list(RISK_FACTORS.keys()))}

#default_types_ent = dmc.Stack(children= [dmc.Stack([dmc.Text(html.H5(layer))] + [make_switch(label, colors_suggested_ent[layer],term) for term, label in layer_suggested_terms.items()]) for layer, layer_suggested_terms in multi_layer_suggested_terms.items()],
#            id="annotate-label-badge-group-ent", align='flex-start', justify = 'flex-start', spacing ='xs')
default_types_ent = make_types_checkbox(group_colors=PALETTE_ent, multi_layer_suggested_terms=multi_layer_suggested_terms)

default_types_fact = dbc.Row([dbc.Col(make_switch(label, colors_suggested_fact[t],t, value=False)) for t, label in FACTUALITY.items()],
            id="annotate-label-badge-group-fact", )

default_types_temporal = dbc.Row([dbc.Col(make_switch(label, colors_suggested_temp[t],t, value=False)) for t, label in TEMPORAL.items()],
            id="annotate-label-badge-group-fact", )

default_types_risk = dbc.Row([dbc.Col(make_switch(label, colors_suggested_risk[t],t, value=False)) for t, label in RISK_FACTORS.items()],
            id="annotate-label-badge-group-risk", )
tooltips = {}

# Init_home_page_content(app_skeleton):
prediction_file_paths, classification_file_paths = get_result_cardio_files(RESULT_PATH_cardio)
#print(len(prediction_file_paths), len(classification_file_paths), 'length of files')
#TEXT_IDS = sorted([k.split('.')[0] for k in prediction_file_paths.keys()])
#sorted(TEXT_IDS)
#print(text_ids, 'the text ids')

login_annotator = html.Div(children = [dbc.Row([ 
                                                dbc.Col(html.H5("Please first login in and then select the document ID."),width=3),
              dbc.Col(dcc.Dropdown(placeholder="Select registered user Email", options = [{'label': u, 'value': u } for u in USER_DB.keys()], id = "annotate-email"),width=2),
        dbc.Col(dbc.Input(placeholder="Password", id = "annotate-password", type= "password"),width=2),
        dbc.Col(dbc.Button('Login', id="annotate-login"), width = 2)], style={'padding':'1rem 1rem'}),
    dbc.Alert("User Email must be given for reseting password! ", id="annotate-missing-email", is_open=False),
    dbc.Alert("Invalid Password. ", id='annotate-sigin-fail-alert', color="danger", is_open=False)]
)

select_section = html.Div([
    dbc.RadioItems(id = "annotate-select-section", options = [])])


#select_document = html.Div(children = [dbc.Label('Document ID', style={'font-size': '20px'}),
#                                       dbc.Row([dbc.Col(dcc.Dropdown(id='annotate-selected-id'), width=8),
#                                                dbc.Col(dbc.Button("Next", id="annotate-next-doc", outline=True, color="primary",  size="sm")), 
#                                                dbc.Col(dbc.Alert("The last document!",id="alert-fade",dismissable=True,is_open=False))])])
# Row 2
#section_checklist =  html.Div(children = [dbc.Label('Select sections for revision', style={'font-size': '20px'}), 
#                                          dbc.Checklist(id='annotate-cardio-section-checklist', options = [{'label': s, 'value':s} for s in list(set(CLASSIFICATION_DB['Section']))], value = [], label_style={'font-size': '20px'})])
# Row 3 data statistic
number_of_tokens = []
number_of_sentences = []
plot_data_statistic = html.Div(children = [dbc.Label('Number of sentences and tokens', style={'font-size': '20px'}), html.Div(id="annotate-table-number-of-sents-tokens")])
 
html_div_children = []
html_div_dropdown_ids = {}

#left_container_children = [
#                        select_document, html.Hr(),
#                        html.Br(), 
#                         plot_data_statistic,
#                            html.Br(),
#                                 html.Hr(style={'border-top':'1px solid blue'}),
                                 
#                                 ]
section_progresses = {"Diagnosen": 155, "Befunde": 157, "Zusammenfassung" :147} 
progress_div = html.Div(children = [
                                    dbc.ListGroup([html.H5(sec), 
                                                   dbc.Progress(id ="annotate-progress-{}".format(sec) ,value=0, label="{}\%".format(0))]) for sec in IDEAL_SECTIONS])
left_container_children = [dbc.Label('Select Section', style={'font-size': '20px'}), 
                        select_section, html.Hr(),
                        html.Br(), 
                        dbc.Label('Progress', style={'font-size': '20px'}), 
                        progress_div ]

left_container =  dmc.Card(id='annotate-left-container',  
                           children=left_container_children, 
                           style = { "overflow":'scroll',
                            "background-color": "#f8f9fa", "height": "520px"}, withBorder=True,
                            shadow="sm",
                        radius="md")

vis_head_row = dbc.Label("Annotation Visualization by Semantic Groups", style={'font-size': '25px'}, color="primary")
vis_groups =html.Div(children = [
    dmc.Tooltip([dbc.Switch(
        id="annotate-vis-group-{}".format(g), 
        label=g, 
        label_style={'color': PALETTE_ent.get(g).text.value, 'font-size':'15px'}, 
        value=True)],
        color = PALETTE_ent.get(g).text.value,
        multiline=True,
        width=100,
        label = ' '.join([t + ': ' + w  for t,w in terms.items()]), position="top" ) for g, terms in multi_layer_suggested_terms.items()
    ] ,
            )
#vis_groups = html.Div(
#    children = [dbc.Accordion(
#            [dbc.AccordionItem(
#                    [dmc.Text(t + ': ' + w ) for t,w in terms.items()],
#                    title=g,
#                    id = "annotate-vis-group-{}".format(g),
#                    style={'color': PALETTE_ent.get(g).text.value, 'font-size':'15px'}) 
#             for g, terms in multi_layer_suggested_terms.items()], 
#           flush=True
#            )]
#)

#vis_tabs = dmc.Tabs(id='annotate-container-vis-tabs', 
#                                                    orientation="horizontal", variant='pills',
#                                                    persistence_type='local')
vis_head = button_group = html.Div( 
    [ dbc.Row([dbc.Col(dbc.Label("Document IDs"), width = 1),
               dbc.Col(dbc.RadioItems(
            id="annotate-vis-section-radios",
            className="btn-group",
            inputClassName="btn-check",
            labelClassName="btn btn-outline-primary",
            labelCheckedClassName="active",
            label_style= {'font-size':'18px'}
        ))]),
    ],
    className="radio-group",
)

def_offcanvas = dbc.Offcanvas([html.Div([
    html.P([
        dmc.Text('{}: {}'.format(t, name), style={'color': PALETTE_ent.get(group).text.value, 'font-size':'15px'}),
        dmc.Text("" if t not in TERM_DEF_DB else TERM_DEF_DB[t][1])]) for t, name in terms.items()]) for group,terms in multi_layer_suggested_terms.items()
    ],
    id="annotate-vis-def-offcanvas", backdrop=False, is_open=False)

open_def_offcanvas = dbc.Button("Check type definitions", id ="annotate-vis-open-definition")

vis_classification = html.Div(id="annotate-vis-section-classification", style = {"height": "330px", "overflow": "scroll" })
card_vis = dmc.Card(children=[vis_head, html.Hr(), vis_classification], style = {'borderWidth': '1px',
            'borderRadius': '2px', "background-color": "#ffffff","height": "auto", "overflow": "scroll" }, shadow="sm" )

semantic_group_types = html.Div(children = [
    dbc.RadioItems(id='annotate-select-groups', options=[{'label':g, 'value':g} for g in multi_layer_suggested_terms.keys()],
            className="btn-group",
            inputClassName="btn-check",
            labelClassName="btn btn-outline-warning",
            labelCheckedClassName="active",
            label_style= {'font-size':'15px'},
            value = list(multi_layer_suggested_terms.keys())[0]), 
    html.Hr(),
    dbc.RadioItems(options=[],id='annotate-select-group-types', 
            inline=True,className="btn-group",
            inputClassName="btn-check",
            labelClassName="btn btn-outline-success",
            labelCheckedClassName="active",
                label_style={'font-size':'15px'})], style={ 'borderRadius': '2px',"background-color": "#f8f9fa", "height": "auto"})


# Explanation container
header = dbc.Row(children=[dbc.Col(dmc.Text('Type score:', style={'background-color':'#aed6f1'}, size='l', 
                                                 weight=500,), width=4), dbc.Col(dmc.Text('Tokens score interpretation:', style={'background-color':'#aed6f1'}, weight=500))])
container_exp = dbc.Container(children = [header,html.Hr(),
    dmc.Card(id="annotate-container-exp",  style = {'borderWidth': '1px',
            'borderRadius': '2px', "background-color": "#ffffff","height": "500px", "overflow": "scroll"}, shadow='sm')])

annotate_spans = html.P(id="annotate-container-revise")

dropdowns = {g: {'options': [{'label':'O', 'value': 'O'}] +[{'label':t, 'value': t} for t in terms] } for g, terms in multi_layer_suggested_terms.items()}
annotate_table = dash_table.DataTable( id="annotate-revise-table", editable=True, column_selectable="multi", selected_columns=[],
    style_cell=
        { 'textAlign': 'left', 
         'font-size':'20px',
         'overflow': 'hidden',
        'textOverflow': 'ellipsis',
        'height':'auto',
        'minWidth': '80px', 
        'width': '180px', 
        'maxWidth': '380px',
        'whiteSpace': 'normal'
         
        } 
    ,
    fixed_columns={'headers': True, 'data': 1},
    style_data={
        'color': 'black',
        'backgroundColor': 'white',
         'whiteSpace': 'normal',
        'height': 'auto',
        
         'lineHeight': '15px',
        
    },
    style_header={
        'backgroundColor': 'rgb(210, 210, 210)',
        'color': 'black',
        'fontWeight': 'bold',
        'padding': '1rem',
        'font-size' :'20px'
    },  
    style_table={'minWidth': '98%'},
    style_as_list_view=True,  
    style_cell_conditional = [{
            'if': { 'column_id':  'Semantic Groups'},
            'background_color': '#D2F3FF',
        'width': '280px', 
                }]
        
)

#annotate_table = dbc.Table(id = "annotate-revise-table", bordered=True)
#annotate_table = dash_table.DataTable(id = "annotate-revise-table")
container_ann = html.Div(children = [ dbc.Row(children=[
            dbc.Col(dmc.Text('Select the span by columns and semantic types by groups'),style={'background-color':'#aed6f1', 'textAlign': 'left'}), 
                      dbc.Col(dbc.Button('Reselect columns', id= "annotate-reselect", n_clicks=0,  color="primary", size="md"))]), 
    dmc.Card(children =[
        html.Hr(),
        semantic_group_types,
        html.Hr(),
        annotate_table, 
      ],  
        style = {'borderWidth': '1px',
            'borderRadius': '2px', "background-color": "#ffffff","height": "500px", "overflow": "scroll", }, 
                         shadow="sm" ), 
        dbc.Row([
            dbc.Col([dbc.Pagination(id="annotate-exp-pagination-sentence-id", active_page=1, max_value = 1, previous_next=True, fully_expanded=True)]),
            dbc.Col()]),
        dbc.Row([dbc.Col(dbc.Button('Save', id= "annotate-save", n_clicks=0,  color="success", size="md")),
          dbc.Col(dbc.Alert(id='annotate-alert-save'), width=10)])])
#  dbc.Button('Update', id= "annotate-update-btn", n_clicks=0,  color="primary", size="sm"),
       


container_vis = html.Div(children= [ 
                                     vis_head_row, 
                                      html.Hr(),
                                    dbc.Row(children = [dbc.Col([vis_groups, open_def_offcanvas], width='auto'), dbc.Col([card_vis], width=10)]),
                                    html.Br(),
                                    ])

container_exp_anns = html.Div(children = [html.Hr(style={'border-top':'1px solid blue'}),
                                    dbc.Row([
                                        dbc.Col([dbc.Label("Inspect Semantic Type Scores and Annotate ", style={'font-size': '25px'}, color="primary")]),
                                        ]),
                                    dbc.Row([dbc.Col([container_exp, 
                                                   ], width=4), 
                                              dbc.Col([container_ann, 
                                                   ], width=8)]),], style={'padding':'1rem 1rem'})


task_overview = html.Div([dbc.Button("Task Overview", id="annotate-overview-btn", n_clicks=0, className='mb-3', color= 'primary'), 
                 dbc.Collapse(
                     dmc.Card(id='annotate-eval-task-overview', style={'borderWidth': '1px',
            'borderRadius': '2px', "background-color": "#ffffff","height": "auto", "overflow": "scroll" }, shadow="sm"),
                     id = 'annotate-overview-collapse')])


classification_results = html.Div([dbc.Button("Classification Results", id="annotate-results-btn", n_clicks=0, className='mb-3', color= 'primary'), 
                          dbc.Collapse(
                              dmc.Card(id='annotate-classification-results', style={'borderWidth': '1px',
            'borderRadius': '2px', "background-color": "#ffffff","height": "auto", "overflow": "scroll" }, shadow="sm"),
                              id = "annotate-results-collapse")])


# dbc.Row(children = [dbc.Col(task_overview, width=6),dbc.Col(classification_results, width=6),)
plots =   html.Div(children = [html.Hr(style={'border-top':'2px solid black'}),
                               task_overview,
                               html.Br(),
                               classification_results
                               ],style={'padding':'1rem 1rem'})

store_annotator = dcc.Store(id="annotate-annotator-memory", storage_type="session", data="BERT-SNER")
store_tsv_sentences = dcc.Store(id='annotate-tsv-or-line-sentences-memory', storage_type='session')
store_section_tab = dcc.Store(id='annotate-section-tab-memory', storage_type = 'session')
store_section_sentence = dcc.Store(id='annotate-section-sentence-memory', storage_type='session')
store_section_scores = dcc.Store(id='annotate-section-scores-dict-memory', storage_type='session')
store_sentences_classification = dcc.Store(id="annotate-sentences-classification-memory", storage_type="session")
store_sentences_classification_revision = dcc.Store(id = "annotate-sentences-revision-memory", storage_type= "session")
store_prediction_files = dcc.Store(id="annotate-prediction-files-memory", storage_type="session", data=prediction_file_paths)
store_classification_files = dcc.Store(id="annotate-classification-files-memory", storage_type = "session", data=classification_file_paths)
store_prediction_dict = dcc.Store(id='annotate-prediction-memory', storage_type='session')
store_classification_dict = dcc.Store(id='annotate-classification-memory', storage_type = "session")
store_classification_section_db = dcc.Store(id='annotate-classification-section-memory', storage_type='session')
store_classification_annotator_section_db = dcc.Store(id='annotate-classification-section-annotator-memory', storage_type='session')

store_vis_selected_groups = dcc.Store(id='annotate-vis-selected-groups-memory', storage_type='session')

store_spans_all_sentences = dcc.Store(id='annotate-spans-all-sentences-memory', storage_type='session')
store_word_scores = dcc.Store(id = 'annotate-word-scores-memory', storage_type ='session')

title = html.Div(id='annotate-interface-title')


storage = [store_annotator,
        store_prediction_files,
                    store_classification_files,
                    store_tsv_sentences,
                    store_spans_all_sentences,
                    store_section_tab,
                    store_section_scores,
                    store_sentences_classification,
                    store_sentences_classification_revision,
                    store_word_scores,
                    store_vis_selected_groups,
                    store_section_sentence, 
                    store_prediction_dict,
                    store_classification_dict,
                    store_classification_section_db,
                    store_classification_annotator_section_db]

home_page_content = [login_annotator, html.Hr(), def_offcanvas,
    dbc.Row([dbc.Col([left_container], width=2), 
                        dbc.Col([container_vis], width=10)], style={'padding':'1rem 1rem'}),
                      container_exp_anns,
                     plots]

layout = html.Div(children = storage + home_page_content  , id='annotate-home-page-content')

#@callback(Output("annotate-selected-id", "value"),
#          [Input("annotate-selected-id", "value"),
#           Input("annotate-next-doc", 'n_clicks')],
#          State('annotate-email', 'value'))
#def next_id(current_id, next_btn, annotator):
#    if ctx.triggered_id == "annotate-next-doc":
#        if annotator is not None:
#            annotator_db = USER_DB[annotator]
#            if  current_id is None:
#                current_id = annotator_db.loc[0, 'last_id']
    
#        if current_id is not None:
            
#            current_id_index = TEXT_IDS.index(current_id)
#            current_id_index_next = min(current_id_index+1, len(TEXT_IDS)-1)
#            current_id = TEXT_IDS[current_id_index_next]
#    return current_id



@callback([Output('annotate-select-section', 'options'),
           Output('annotate-password', 'valid'),
           Output('annotate-sigin-fail-alert', 'is_open'),
           Output("annotate-annotator-memory", 'data')],
          [Input('annotate-email', 'value'),
           Input('annotate-password', 'value'),
           Input('annotate-login', 'n_clicks')
          ])
def sign_in(user_email, password, login_btn):
    user_password_valid = False
    login_fail = False
    section_options = []
    if login_btn:
        user_current_db = USER_DB[user_email]
        if password == str(user_current_db['password']):
                user_password_valid = True
                #section_options = [{'label':d, 'value':d} for d in TEXT_IDS]
                section_options = [{'label': sec,  'value': sec} for sec in IDEAL_SECTIONS]
               
        else:
            login_fail = True
                                                    
    return section_options, user_password_valid, login_fail, user_email


@callback(Output('annotate-vis-selected-groups-memory', 'data'),
          [Input("annotate-vis-group-{}".format(g), 'value') for g in list(multi_layer_suggested_terms.keys())]
          )
def update_selected_group(g1, g2, g3, g4, g5, g6, g7):
    """
    dbc.
    Switch(id="annotate-vis-group-{}".format(g), label=g, label_style={'color': PALETTE_ent.get(g).text.value,
    'font-size':'20px'}, value=True) for g in multi_layer_suggested_terms.keys()]
    """
    
    selected_gs = []
    for i, g_value in enumerate([g1,g2,g3,g4,g5, g6, g7]):
        if g_value:
            group = list(multi_layer_suggested_terms.keys())[i]
            if group not in selected_gs:
                selected_gs.append(group)
    return selected_gs

@callback(Output('annotate-select-group-types', 'options'),
          Input('annotate-select-groups', 'value'))
def return_label_types(group_name):
    options = []
    #label_style = {}
    value = []
    if group_name:
            #label_style = {'color': PALETTE_ent.get(group_name).text.value} 
            options = [{'label': t+': '+ w, 'value': t} for t, w in multi_layer_suggested_terms[group_name].items()] + [{'label': 'O: null', 'value': 'O'}]
            #value = [t for t, w in multi_layer_suggested_terms[group_name].items()]
    return options

@callback([Output('annotate-vis-section-radios', 'options'),
            Output('annotate-vis-section-radios', 'value')],
          Input('annotate-select-section', 'value'),
          State('annotate-annotator-memory', 'data'))
def make_ids_radios(section, annotator):
    options = []
    if section is not None:
        options = [{'label': d, 'value': d } for d in IDEAL_SECTIONS_IDS[section]]
    return options, None
        
@callback([
            Output('annotate-tsv-or-line-sentences-memory', 'data'),
            Output('annotate-prediction-memory', 'data'),
            Output('annotate-classification-memory', 'data')], 
        Input('annotate-vis-section-radios', 'value'),
         [State('annotate-select-section', 'value'),
             State('annotate-annotator-memory', 'data'),
          ])
def get_cardio_tsv_sentences(text_id, section, annotator):
    sentences_dict = {}
    prediction_dict = {}
    classification_dict = {}
    #selected_sec_default = None
    
    number_of_sents_tokens_table = []
    number_of_sents_tokens_list = []
    
    section_radios_options = []
    #print('prediction file paths ', prediction_file_paths)
    #print('classification file paths', classification_file_paths)
    for key in prediction_file_paths:
        if key not in classification_file_paths:
            print(key, ' not in classification file path ')

    if text_id is not None:
        prediction_dict, classification_dict_original = return_json_cardio_dict(text_id, RESULT_PATH_cardio, prediction_file_paths=prediction_file_paths, classification_file_paths=classification_file_paths)
        #section_options, section_values = make_section_options(list(prediction_dict.keys()))
        #section_radios_options = [{'label' : s, 'value' :s} for s in prediction_dict.keys()]
        
        #for section, section_classification in classification_dict_original.items():
        if section in classification_dict_original: 
            section_classification = classification_dict_original[section]
            classification_dict[section] = {}
            for group in section_classification.keys():
                if group == "Zeitliche Information":
                    classification_dict[section]['Temporal'] = section_classification[group][THRESHOLD]
                else:
                    classification_dict[section][group] = section_classification[group][THRESHOLD]
            
        tsv_lines = get_tsv_lines(text_id)
        sentences_dict = retrieve_sentences(tsv_lines)
        
        
        #for sec in section_radios_options:
            #print(sec)
        #    sentence_ids = prediction_dict[sec['label']]['sentence_ids']
        #    sentences, number_of_sentences, number_of_tokens = get_section_sentences(sentence_ids, sentences_dict)
        #    number_of_sents_tokens_list.append({'Section': sec['label'], 'NumSentences': number_of_sentences, 'NumTokens' : number_of_tokens})
        #df = pd.DataFrame(number_of_sents_tokens_list)
        #number_of_sents_tokens_table = dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns], style_cell={'textAlign': 'left'}, style_table={ 'overflowX': 'auto'} )
        #selected_sec_default = list(prediction_dict.keys())[0]
    return sentences_dict, prediction_dict, classification_dict
        
@callback(Output("annotate-sentences-classification-memory", 'data'),
          [
           Input('annotate-vis-section-radios', 'value'),
         Input('annotate-vis-selected-groups-memory', 'data'),
         Input('annotate-classification-memory', 'data'),
         Input("annotate-sentences-classification-memory", 'data')],
          [State('annotate-select-section', 'value'), 
              State("annotate-annotator-memory", 'data')])
def get_group_sentences_classification(tsv_id, selected_groups, classification_dict, sentences_classification_dict, section, annotator):
   
    if sentences_classification_dict is None:
         sentences_classification_dict = {}
        
    if  section is not None and section in classification_dict:
        if section not in sentences_classification_dict:
            sentences_classification_dict[section] = {}
            
        path_annotator = os.path.join(RESULT_PATH_cardio, "classification_{}.json".format(annotator))
        if tsv_id is not None and selected_groups is not None: 
            if os.path.exists(path_annotator):    
                CLASSIFICATION_DB_ANNOTATOR = json.load(open(path_annotator))
                if section not in CLASSIFICATION_DB_ANNOTATOR:
                    sentences_classification_dict[section][tsv_id] = {group : classification_dict[section][group] for group in selected_groups}
                else:
                    if tsv_id in CLASSIFICATION_DB_ANNOTATOR[section]:
                        sentences_classification_dict[section][tsv_id] = CLASSIFICATION_DB_ANNOTATOR[section][tsv_id]
                    else:
                        sentences_classification_dict[section][tsv_id] = {group : classification_dict[section][group] for group in selected_groups}
            else:
                sentences_classification_dict[section][tsv_id] = {group : classification_dict[section][group] for group in selected_groups}
        
    #print(tsv_id, section)
    #print("sentences_classification_group ", sentences_classification_dict.keys() )
    return sentences_classification_dict

@callback(Output("annotate-vis-def-offcanvas", "is_open"),
          Input("annotate-vis-open-definition", 'n_clicks'))
def open_definition_offcanvas(open_btn):
    open_offcanvas = False
    if ctx.triggered_id == "annotate-vis-open-definition":
        open_offcanvas = True
    return open_offcanvas
       
@callback([Output("annotate-vis-section-classification", 'children'),
           Output('annotate-section-scores-dict-memory','data')], 
        [
            Input("annotate-sentences-classification-memory", 'data'),
         Input("annotate-sentences-revision-memory", 'data'),
         Input('annotate-vis-section-radios', 'value'), 
         Input('annotate-vis-selected-groups-memory', 'data'),
         Input('annotate-tsv-or-line-sentences-memory', 'data'),
         Input('annotate-prediction-memory', 'data')
         ], 
        [State('annotate-select-section', 'value'),
            State("annotate-annotator-memory", 'data')])
def make_section_vis( group_sentences_classification_dict, group_sentences_revision_dict, tsv_id, selected_groups, sentences_dict, prediction_current, section, annotator):
        vis_classification = []
        spans_all_sentences = []
        selected_types = []
        scores_dict_sec = {}
        #print('prediction current ', prediction_current.keys())
        #print('group sentences classification dict ', group_sentences_classification_dict)
        if section in group_sentences_classification_dict and tsv_id in group_sentences_classification_dict[section]:
            if group_sentences_revision_dict is not None and section in group_sentences_revision_dict:
                #print('revision dict ', group_sentences_revision_dict)
                if tsv_id in group_sentences_revision_dict[section]:
                    current_sec_sentences_classification_dict = group_sentences_revision_dict[section][tsv_id]
                else:
                     current_sec_sentences_classification_dict = group_sentences_classification_dict[section][tsv_id]
            else:
                #print("memory dict", group_sentences_classification_dict)
                current_sec_sentences_classification_dict = group_sentences_classification_dict[section][tsv_id]
            
            
            if selected_groups is not None and len(selected_groups) > 0:
                 for group in selected_groups:  
                    selected_types.extend(list(multi_layer_suggested_terms[group].keys()))
        
        
            sentence_ids, scores_dict_sec  = get_prediction_section(prediction_current, section)
            spans_all_sentences = get_spans_all_sentences(sentence_ids, sentences_dict)
            if len(selected_types) > 0:
                
                for group in selected_groups:
                    
                    _ = get_sentences_spans_ner(current_sec_sentences_classification_dict[group], spans_all_sentences= spans_all_sentences, selected_sem_types=selected_types)
                
                lines_str_html_ths = ''
                for t, sent_spans in spans_all_sentences.items():
                    lines_str_html = format_span_line_markup(text=sent_spans['text'], spans=sent_spans['ner_spans'], palette=PALETTE_types)
                    lines_str_html_ths += '<hr>{} {}</hr>'.format(int(t) +1, ''.join(lines_str_html))
                vis_classification.append(Purify(lines_str_html_ths))
            #one_tab = make_one_section_ner_tab('tab-{}'.format(i), lines_str_html_ths)
             #   tabs.append(one_tab)
            #    section_tab['tab-{}'.format(i)] = sec
        return vis_classification, scores_dict_sec
    
@callback([Output('annotate-word-scores-memory', 'data'),
           Output("annotate-exp-pagination-sentence-id", "max_value")],
                [Input('annotate-prediction-memory', 'data'), 
                 Input('annotate-tsv-or-line-sentences-memory', 'data'), 
                 Input('annotate-select-groups', 'value'),
                 ],
                State('annotate-select-section', 'value'))
def get_sentences_for_explanation(scores_dict, tsv_sentences, group, section):
        
        main_children = [] 
        word_scores = {}
        sem_types = []
        len_sentences = 1
        #if len(term_types) > 0:
        sem_types = list(multi_layer_suggested_terms[group].values())

        if len(sem_types) > 0 and scores_dict is not None and section in scores_dict:
                #print(scores_dict.keys())
                
                scores_dict_sec, sentence_ids = get_scores_dict_by_sec(scores_dict, section)
                word_scores = get_sentence_scores_by_type_cardio(scores_dict_sec,  sentence_ids=sentence_ids, tsv_sentences=tsv_sentences, sem_types=sem_types, ca=group)
              
                len_sentences = len(sentence_ids)
        return word_scores, len_sentences
    
@callback(Output("annotate-container-exp", 'children'),
              [Input("annotate-exp-pagination-sentence-id", 'active_page'),
              Input('annotate-word-scores-memory', 'data')])
def make_sentence_explanation(active_page, word_scores):   
        #if len(sem_types) > 0 and 
        
            #main_children.append(header)
        spoiler_children = []
        if word_scores is not None and len(word_scores) > 0: 
            sem_types = list(word_scores.keys())
            #print(sem_types)
  
            for ty, current_sent_scores in word_scores.items():
                if current_sent_scores is not None and len(current_sent_scores) >= active_page:
                    current_sent_score = word_scores[ty][active_page-1]
                    type_p = dash_html_prediction_score(current_sent_score['type_scores'][0],current_sent_score['type_scores'][1], cmap_name='bwr' )
                    word_p = dash_html_prediction_score(current_sent_score['word_scores'][0],current_sent_score['word_scores'][1], cmap_name='bwr' )
                    spoiler_children.append(dbc.Row(children=[dbc.Col(dmc.Text(type_p),width=4),dbc.Col(dmc.Text(word_p, style={'overflow-x': 'auto',}))]))
                    spoiler_children.append(html.Hr())
               # main_children.append(dmc.Spoiler(showLabel="Show more",hideLabel="Hide",maxHeight=200, children = spoiler_children))
        return [dmc.Spoiler(children = spoiler_children, showLabel="Show more",hideLabel="Hide", maxHeight=500)]


@callback([Output("annotate-sentences-revision-memory", 'data'),
         Output("annotate-revise-table", 'data'),
           Output("annotate-revise-table", 'columns'),
           Output("annotate-revise-table", 'selected_columns'),
           Output("annotate-revise-table", 'style_data_conditional'),
           Output('annotate-select-group-types', 'value'),
           Output("annotate-exp-pagination-sentence-id", 'active_page')],
        [Input("annotate-sentences-revision-memory", 'data'),
        Input("annotate-sentences-classification-memory", 'data'), 
        Input("annotate-exp-pagination-sentence-id", 'active_page'),
        
        Input('annotate-vis-section-radios', 'value'),
        Input("annotate-revise-table", 'selected_columns'),
        
        Input("annotate-reselect", 'n_clicks'),         
        Input('annotate-select-group-types', 'value'),
        Input('annotate-select-groups', 'value')],
        State('annotate-select-section', 'value'))
def make_sentence_annotation(revised_memory, group_sentences_classification_dict, active_page, tsv_id,  selected_columns, reselect_btn, type_selected, group_selected, section):   
        #if len(sem_types) > 0 and 
        #print(section, 'section')
        if revised_memory is None:
            revised_memory = {}
        else:
            if section not in revised_memory and  section in group_sentences_classification_dict:
        #ctx.triggered_id == "annotate-vis-section-radios":
            #if len(revised_memory) == 0 and len(group_sentences_classification_dict) > 0: 
                #print(ctx.triggered_id)
                #print(tsv_id, section)
                #if  tsv_id in group_sentences_classification_dict[section]:
                revised_memory[section] = deepcopy(group_sentences_classification_dict[section])
        
        if ctx.triggered_id == "annotate-reselect":
                selected_columns = []
                #print(ctx.triggered_id, selected_columns)
        
        #header = [html.Th("Token")]+ [html.Th(group) for group in multi_layer_suggested_terms.keys()]
        data = {'Token': []}
        groups = list(multi_layer_suggested_terms.keys())
        data.update({group: [] for group in groups})
        #all_options = { group: [{'label' : 'O', 'value': 'O'}] + [{'label': term, 'value': term} for term in term_types_dict.keys()] for group, term_types_dict in multi_layer_suggested_terms.items()}
        #rows_tokens = []
        #rows_groups = {g: [] for g in multi_layer_suggested_terms.keys()}
        current_page = 0
        #print('revised memory for annotation', revised_memory.keys(), 'check')
        if revised_memory is not None and len(revised_memory) > 0 and section in revised_memory: 
            if tsv_id in group_sentences_classification_dict[section] and tsv_id not in revised_memory[section]:
                revised_memory[section][tsv_id] = deepcopy(group_sentences_classification_dict[section][tsv_id])
            
            if ctx.triggered_id == "annotate-select-group-types" and type_selected is not None:
                    #print(ctx.triggered_id, "annotation ")
                    #print(group_selected, type_selected, active_page, )
                    #print("type selected columns and len(selected columns)", type(selected_columns), len(selected_columns))
                    if type_selected is not None and len(selected_columns) >= 1:
                        #print("type selected ", type_selected, "column selected ", selected_columns[0])
                        if len(revised_memory[section][tsv_id][group_selected]) >= active_page-1:
                            #print("current page ", active_page)
                            for sel_col in selected_columns:
                                revised_memory[section][tsv_id][group_selected][active_page-1][int(sel_col)][2] = type_selected
                                #print("changed revised_memory ")
            
            #print('revised memory for annotation', revised_memory.keys())
            
            if tsv_id in revised_memory[section]:
                for  group, sentences_classification in revised_memory[section][tsv_id].items():
                    #print(sentences_classification)
                    #sentences_classification = revised_memory[][group]
             
                    if active_page is not None and len(sentences_classification) >= active_page:
                        current_page = active_page-1
            
                #if current_page not in sentences_classification:
                #        current_page = str(current_page)
            
                #if len(sentences_classification) > 0:
                        #print(current_page, len(sentences_classification))
                        sentence_classification = sentences_classification[current_page]
                        
                        for idx, triple in enumerate(sentence_classification):
                            #if len(sentence_classification) == 1:
                            #    print(idx, triple, data['Token'])
                            if len(data['Token']) < len(sentence_classification):
                                data['Token'].append(triple[0])
                            
                            data[group].append(triple[2])
                        
                        #if len(sentence_classification) == 1:
                        #    print(sentence_classification)
                        #    print(data['Token'])
            
                missing_groups = list(set(multi_layer_suggested_terms.keys()) - set(group_sentences_classification_dict[section][tsv_id].keys()))
                if len(missing_groups) > 0:
                    for g in missing_groups:
                        data[g] = ['O'] * len(data['Token'])            
        
        
            
        #current_table_df = pd.DataFrame(
        #            OrderedDict([(col, value_list) for col, value_list in data.items()]     
        #        ))
        
        #header = [html.Thead(html.Tr([html.Th()] + [html.Th(t+ " " * (10-len(t))) for t in data['Token']]))]
        # [{'id': c, 'name': c} for c in df.columns]
        columns = [{'id': 'Semantic Groups', 'name':  'Semantic Groups'}] + [{'id': str(t), 'name': w, "selectable": True, 'type': 'text'} for t, w in enumerate(data['Token'])]
        
        data_list =  [('Semantic Groups', groups)] 
        
        for i, t in enumerate(data['Token']):
                current_row = (str(i), [])
                for group in groups:
                    current_row[1].append(data[group][i])
                data_list.append(current_row)
                
        #print(columns)
        #print(data_list)
        
            #print(len(columns), columns)
            #print(data_list)
        data_dict = OrderedDict(data_list)
        data_df = pd.DataFrame(data_dict)
        #print(data_df.to_dict('records'))
        
        style_selected_columns = [{
            'if': { 'column_id': i },
            'background_color': '#D2F3FF'
                } for i in selected_columns]
        
        
        data_df['id'] = data_df.index
       
        return revised_memory, data_df.to_dict('records'), columns, selected_columns, style_selected_columns, None, active_page

@callback([Output("annotate-alert-save", "is_open"),
           Output("annotate-alert-save", 'children')],
          [Input("annotate-save", 'n_clicks'),
           Input("annotate-sentences-revision-memory", 'data'),
            Input('annotate-vis-section-radios', 'value')], 
            [State("annotate-annotator-memory", 'data'),
            State('annotate-select-section', 'value')])
def make_revision(n_clicks, revise_memory,  tsv_id, annotator, section):
    text = ""
    save_btn = False
    last_save_id = None
    
    #if ctx.triggered_id == "annotate-save" and tsv_id is not None:
    if n_clicks and tsv_id is not None:
        if annotator is not  None and revise_memory is not None and len(revise_memory) > 0:
            current_revision_dict = {section : {
                    tsv_id: revise_memory[section][tsv_id]
                }
                    }
            
            path_anntotator = os.path.join(RESULT_PATH_cardio, "classification_{}.json".format(annotator))
            if not os.path.exists(path_anntotator):
                #json.dump(current_revision_dict, open(path_anntotator, 'w'), indent=4)
                dict_str = json.dumps(current_revision_dict, indent=2)
                with open(path_anntotator, 'w') as outfile:
                    outfile.write(dict_str)
            else:
                previous_saved = json.load(open(path_anntotator))
                #print(len(previous_saved), "before append ")
                if section in previous_saved:
                    previous_saved[section][tsv_id] = revise_memory[section][tsv_id]
 
                else:
                    previous_saved.update(current_revision_dict)
                    
                #print(len(previous_saved), "after append ")
                dict_str = json.dumps(previous_saved, indent=2)
                #json.dump(previous_saved, open(path_anntotator, 'w'), indent=4)
                with open(path_anntotator, 'w') as outfile:
                    outfile.write(dict_str)
            # columns = []'ID', 'Section', 'Group', 'Annotator', 'Sentence_idx', 'Classification'
                text = "Save revision of section_{} document_{} to local file for current annotator {} .".format(section, tsv_id,  annotator)
                #print(text)
                save_btn = True
            
    return save_btn , text
                   

@callback([
    Output("annotate-progress-{}".format(sec), "value") for sec in IDEAL_SECTIONS] + 
    [Output("annotate-progress-{}".format(sec), "label") for sec in IDEAL_SECTIONS], 
    [Input("annotate-save", 'n_clicks'),
     Input('annotate-login', 'n_clicks'),
     Input("annotate-annotator-memory", 'data')])
def update_progress(save_btn, login, annotator):
    sections = list(section_progresses.keys())
    section_full = {"Diagnosen": 155, "Befunde": 157, "Zusammenfassung" :147} 
    new_value = [0, 0, 0]
    
    labels = [str(0), str(0), str(0)]
    
    if annotator is not None:
        if ctx.triggered_id == "annotate-login" or ctx.triggered_id == "annotate-save":
            path_anntotator = os.path.join(RESULT_PATH_cardio, "classification_{}.json".format(annotator))
            #if ctx.triggered_id == "annotate-save":
            #path_anntotator = os.path.join(RESULT_PATH_cardio, "classification_{}.json".format(annotator))
            if  os.path.exists(path_anntotator):
                previous_saved = json.load(open(path_anntotator,encoding="latin-1"))
                #for tid, sections_saved in previous_saved.items():
                for i, sec in enumerate(IDEAL_SECTIONS):
                    if sec in previous_saved:
                        tid_saved = previous_saved[sec]
                        for tid, terms in tid_saved.items():
                            len_sentences = len(list(terms.values())[0])
                            new_value[i] += len_sentences
                            #print(new_value, 'progress of section ', sec)
        
        
    new_value = [v/150*100 for i, v in enumerate(new_value)]
    labels = [str(ratio)[:4]+"%" for ratio in new_value]
    return new_value[0], new_value[1], new_value[2], labels[0], labels[1], labels[2]
            
            
@callback(Output('annotate-eval-task-overview', 'children'),
          [Input("annotate-save", 'n_clicks'),
           Input("annotate-annotator-memory", 'data')])      
def plot_task_overview(save_btn, annotator):
    
    path_annotator = os.path.join(RESULT_PATH_cardio, "classification_{}.json".format(annotator))
    if os.path.exists(path_annotator):
        classification_annotator = json.load(open(path_annotator, encoding='latin-1'))
    else:
        classification_annotator = {}
    section_sentences = get_section_sentences_dict(RESULT_PATH_cardio, prediction_file_paths, classification_file_paths)
    amount_sentences = get_amount_sentences(section_sentences=section_sentences, classification_annotator=classification_annotator, annotator=annotator)
    amount_annotation = get_amount_annotations(section_sentences=section_sentences, classification_annotator=classification_annotator, annotator=annotator)
    
    amount_df = pd.DataFrame(amount_sentences)
    annotation_df = pd.DataFrame(amount_annotation)
    fig_amount_df = px.bar(amount_df, x = 'Section', y= 'Sentence Amount', color = "Annotator", barmode="group" ,title="Amount of Sentences of Selected Documents (Annotated by Model BERT-SNER) and Reviewed by Current Annotator {}".format(annotator))
    fig_annotation_df = px.bar(annotation_df, x = "Group", y= "Amount", color="Annotator", barmode = "group",
            facet_col = "Section", title="Amount of Annotations (at token level) by different Semantic Groups of different Sections")
    return [dbc.Row([dbc.Col(dcc.Graph(figure = fig_amount_df)), dbc.Col(dcc.Graph(figure = fig_annotation_df))])]

@callback(Output("annotate-overview-collapse", 'is_open'),
          Input("annotate-overview-btn", 'n_clicks'),
          State("annotate-overview-collapse", 'is_open'))
def toggle_overview(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open

@callback(Output("annotate-results-collapse", 'is_open'),
          Input("annotate-results-btn", 'n_clicks'),
          State("annotate-results-collapse", 'is_open'))
def toggle_overview(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open


@callback(Output('annotate-classification-results', 'children'),
          [Input("annotate-save", 'n_clicks'),
           Input("annotate-annotator-memory", 'data')])
def toggle_classification_results(save_btn, annotator):
    path_annotator = os.path.join(RESULT_PATH_cardio, "classification_{}.json".format(annotator))
    if os.path.exists(path_annotator):
        classification_annotator = json.load(open(path_annotator, encoding='latin-1'))
    else:
        classification_annotator = {}
    
    predictions, annotations_ref = get_y_pred_true(classification_annotator, RESULT_PATH_cardio, prediction_file_paths, classification_file_paths)
    group_reports = get_group_classification_report(predictions=predictions, annotations_ref=annotations_ref)
    
    rows = []
    row = None
    children = []
    header = None
    body = None
    for i, group in enumerate(list(multi_layer_suggested_terms.keys())):

        current_row_label = dbc.Label(group)
        if group in group_reports:
            #print(group)
            header, body = make_report_table(group_reports[group])
            
            row = dbc.Col([current_row_label,html.Hr(), dmc.Table(header + body), html.Hr()])
        
        if row:
            #if len(rows) < 4:
            rows.append(row)
            
        if len(rows) == 4:
            children.append(dbc.Row(rows))
            children.append(html.Br())
            rows = []
                
    if len(rows) > 0:
            children.append(dbc.Row(rows))
    return children
            
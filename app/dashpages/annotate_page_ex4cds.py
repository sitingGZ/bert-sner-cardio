import copy
import dash
from dash import dash_table

from dash import no_update
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc

from dash import dcc, html, callback, clientside_callback

from dash import MATCH, ALL
from dash.dependencies import Output, Input, State
from dash_extensions import Purify

from flask import Flask
import pathlib
import pandas as pd
import json 
import os
import logging
import matplotlib.pyplot as plt
import itertools

from dashpages.utils import make_switch, get_prediction_classification_section, get_spans_all_sentences, get_sentence_classification
from dashpages.utils import get_ex4cds_text_id, get_ex4cds_prediction, get_ex4cds_text_lines, map_token_spans, get_spans_all_sentences_ex4cds
from dashpages.format_markup import dash_html_prediction_score
from dashpages.span_markup import format_span_line_markup

from dashpages.format_markup import Palette, colors_ent, color_fact, color_temporal,color_risk_att

from copy import deepcopy
import torch
from transformers import  AutoTokenizer, AutoModelForMaskedLM

#app =  Dash(__name__, requests_pathname_prefix = '/',meta_tags=[{ "content": "width=device-width, initial-scale=1"}], 
#           external_stylesheets=[dbc.themes.CERULEAN, dbc.icons.FONT_AWESOME], suppress_callback_exceptions=True)

#app.title= "Clinical Text Semantic Analytics"
#server = app.server
#app.config.suppress_callback_exceptions = True

# Preparing the model to interact with the user interface
BASE_PATH = pathlib.Path(__file__).parent.resolve()
#DATA_PATH = BASE_PATH.joinpath("cardiode").resolve()

EX4CDS_PATH = BASE_PATH.joinpath('Ex4CDS_finalAnnotation/finalAnnotation/2').resolve()
RESULT_PATH_ex4cds = BASE_PATH.joinpath('ex4cds_results').resolve()

THRESHOLDS = {'0.5':'threshold_05', '0.6':'threshold_06', '0.7':'threshold_07'}

ANNOTATEPATH = {'span': 'Nephrologie', 'href':'/annotate_ex4cds'}

#Ex4CDS_TYPES_ENTITY = {'Condition': 'Zeichen oder Symptom',
# 'DiagLab': 'Diagnostisch und Laborverfahren',
# 'LabValues': 'Klinisches Attribut',
# 'HealthState': 'Gesunder Zustand',
# 'Medication': 'Pharmakologische Substanz',
# 'Process': 'Physiologische Funktion',
# }

#Ex4CDS_TYPES_KONZEPT = {
# 'Measure': 'Quantitatives Konzept',
# 'TimeInfo': 'Zeitliches Konzept'}

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
                               'T047': "Krankheiten", 
                               'T048': "Psychische oder Verhaltensstoerung"},
                               "Health State":{'T300': "Gesunder Zustand" ,  
                                               "T301": "Verschlechter Zustand"}}



FACTUALITY = {'T400': 'kein oder negiert', 'T401' : 'gering',  'T402' : 'fraglich','T403': 'zukueftig', 'T404': 'unwahrscheinlich'}

TEMPORAL = {'T500': 'aktuelles Ereignis', 'T501': 'Vergangenheit zum aktuellen Ereignis','T502': 'vergangenes Ereignis','T503': 'zukuenftiges Ereignis'}

RISK_FACTORS = {'increase':'Risiko erhöhen', 'decrease': 'Risiko verringern', 'not relevant': 'kein Risiko'}


#SUGGESTED_TYPES_ENT = list(Ex4CDS_TYPES_ENTITY.values())
#SUGGESTED_TYPES_ATT = list(Ex4CDS_TYPES_KONZEPT.values())
SECTION_LIST = ['Anrede', 'AktuellDiagnosen', 'Diagnosen', 'AllergienUnverträglichkeitenRisiken', 'Anamnese', 'AufnahmeMedikation', 'KUBefunde', 'Labor', 'Befunde', 'EchoBefunde', 'Zusammenfassung', 'EntlassMedikation', 'Abschluss']
SECTIONS_NEPH =  ['Rejection', 'Transplantatversagen', 'Infection']

# Build homepage
#def get_text_sections(text_id):
#    sections = [f.split('.')[2] for f in os.listdir(DATA_PATH) if f.split('.') ]
#    return sections

#def get_text_sec_sentences(text_id, section):
#    lines = [l for l in open(DATA_PATH.joinpath('{}.tsv.{}.txt'.format(text_id, section)))]
#    sentences_split = [l.strip().split() for l in lines]
#    return lines, sentences_split


def get_result_ex4cds_files(result_path):
    prediction_file_paths = {'_'.join(f.split('.')[0].split('_')[:-2]): f for f in os.listdir(result_path) if '_score' in f}
    classification_file_paths = {'_'.join(f.split('.')[0].split('_')[:-2]): f for f in os.listdir(result_path) if 'ner_' in f}
    return prediction_file_paths, classification_file_paths


def return_json_ex4cds_dict(text_id,result_path, prediction_file_paths, classification_file_paths):
    prediction_dict = json.load(open(os.path.join(result_path, prediction_file_paths[text_id])))
    classification_dict = json.load(open(os.path.join(result_path, classification_file_paths[text_id])))
    tokens = [[w[0] for w in l ] for l in classification_dict['Physical Object']['0.5']]
    return prediction_dict, classification_dict, tokens


def get_sentence_scores_by_type_ex4cds(scores_dict, spans_all_sentences, sem_types):
    """
    :param sem_types: selected types in all category [type1, type2])
    
    """
    scores_dict_sec = deepcopy(scores_dict)
    #print(scores_dict_sec)
    #sentence_ids = scores_dict_sec['sentence_ids']
    #category, types = sem_types[0], sem_types[1]
    scores_types = {}
    #for ca , tys in sem_types.items():
    #print('spans all sentences in current get sentence scores by type ex4cds ', spans_all_sentences)
    for ty in sem_types:
        for ca, current_scores in scores_dict_sec.items():
            if type(current_scores) == dict:
                #print(current_scores)
                if ty in current_scores: 
                    scores = current_scores[ty]
                    print()
                    ty_words = ty.split()
                    scores_types[ty] = []
                    for i, current_sent in spans_all_sentences.items():
                        
                        #current_sent = tsv_sentences[d]
                        words = [l[2] for l in current_sent['text_spans']]
                
                        ty_scores = [float(s) for s in scores[int(i)][:len(ty_words)]]
                        word_scores = [float(s) for s in scores[int(i)][len(ty_words):]]
                        #print(len(ty_scores), len(word_scores), len(words), len(scores[i]))
                        scores_types[ty].append({'type_scores': (ty_words, ty_scores), 'word_scores': (words, word_scores)})
    return scores_types

def make_section_options(domain, sections_useful = None):
    # CLINIC_DOMAIN = {'Cardiology':'Kardilogie', 'Nephrology':'Nephrologie'}
    if domain == 'Nephrologie':
        return [{'label': s, 'value': s} for s in SECTIONS_NEPH]
    else:
        assert sections_useful is not None, "Sections for domain {} is required.".format(sections_useful)
        #return list(prediction_json_dict.keys())
        return [{'label':s, 'value':s} for s in sections_useful]
    
def get_section_sentences(sec_sentences_ids, tsv_sentences_dict):
    #ids = [d for d in sec_sentences_ids]
    sentences = []
    for d in sec_sentences_ids:
        sentences.append(' '.join([l[2] for l in tsv_sentences_dict[d]]))
    return sentences

def make_one_section_plain_tab( tab_value, sentences):
    textarea = dmc.Text(children= [html.P(s) for s in sentences], style = {'borderWidth': '1px',
        'borderStyle': 'dotted',"height": "auto",
        'borderRadius': '2px',"padding": "1rem 1rem", "background-color": "#ffffff", "overflow": "scroll"})
    return dmc.TabsPanel(textarea, value = tab_value)

def make_one_section_ner_tab(tab_value, html_to_purify):   
    vis_box_children = Purify(html_to_purify)
    vis_box = dmc.Text(style={'white-space': 'pre-wrap', 'padding':'1rem 1rem', "height": "450px",'overflow':'scroll'}, children = vis_box_children, )
    #print(type(vis_box_children), type(vis_box))
    return dmc.TabsPanel(vis_box, value = tab_value)

##def make_one_section_ner_tab(sec_title, classification_results):
#    return 
def make_types_checkbox(group_colors, multi_layer_suggested_terms):
    semantic_types_checkbox_groups = {}
    for group, types in multi_layer_suggested_terms.items():
        semantic_types_checkbox_groups[group] =  dbc.Checklist(
            options=[{'label': t + ': ' + name, 'value':t} for t, name in types.items()
            ],
            label_style = {'color': group_colors.get(group).line.value} ,   
            value=list(types.keys()),
            id="ex4cds-{}-group".format('-'.join(group.lower().split())),
            )
    return semantic_types_checkbox_groups

def make_select_types_to_explain(sec_scores_dict):
    categories = [list(value.keys()) for ca, value in sec_scores_dict.items() if type(value) == dict]
    categories_flat = list(itertools.chain.from_iterable(categories))
    return dcc.Dropdown(id='ex4cds-types-options', options = [{'label':ty, 'value':ty} for ty in categories_flat], multi=True)
                      
# LAYOUT
# Make_left_container   
# Row 1
#default_list = ['AktuellDiagnosen', 'Diagnosen','Zusammenfassung'] 



#text_id_sections_df = pd.DataFrame([{'id':f.split('.')[0], 'section':f.split('.')[2] } for f in os.listdir(DATA_PATH) ])
CLINIC_DOMAIN = {'Cardiology':'Kardiologie', 'Nephrology':'Nephrologie'}

select_domain = html.Div([
    html.H4('Domain'), 
    dcc.Dropdown([d for d in CLINIC_DOMAIN.values()], id='ex4cds-select-domain', multi=False)
    ], style= {'textAlign': 'left'})

#select_document = html.Div([html.H5('Document ID'),dcc.Dropdown([], id="ex4cds-annotate-select-document-id", multi=False)],style= {'textAlign': 'left'})    
# Make_right_container
# NER visualization container
semantic_type_row = dbc.Row([dbc.Col([dcc.Dropdown(id='ex4cds-select-groups', options=[{'label':g, 'value':g} for g in multi_layer_suggested_terms.keys()], multi=False)], width=2), 
                             dbc.Col([dbc.Checklist(options=[], id='ex4cds-select-group-types', inline=True)], width=8)])


# Explanation container
container_exp = dmc.Card(id="ex4cds-container-exp",  style = {'borderWidth': '1px','borderStyle': 'dotted',
            'borderRadius': '2px', "background-color": "#ffffff","height": "auto", "overflow": "scroll" }, shadow="sm" )
container_ann = dmc.Card(id="ex4cds-container-ann",  style = {'borderWidth': '1px','borderStyle': 'dotted',
            'borderRadius': '2px', "background-color": "#ffffff","height": "auto", "overflow": "scroll" }, shadow="sm" )
        
#selected_types = html.Div(id='ex4cds-select-types-explain', children =  dcc.Dropdown(id='ex4cds-types-options'))
#dcc.Dropdown(id='ex4cds-select-types-explain', options = [{'label':label, 'value':label} for label in SUGGESTED_TYPES_ENT + SUGGESTED_TYPES_ATT], value = SUGGESTED_TYPES_ENT[0], multi=False)

                                #dbc.Col([selected_types])])
                                

# Third column
PALETTE_ent = Palette([], {l: colors_ent[i] for i, l in enumerate(list(multi_layer_suggested_terms.keys()))})
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
#            id="ex4cds-label-badge-group-ent", align='flex-start', justify = 'flex-start', spacing ='xs')
default_types_ent = make_types_checkbox(group_colors=PALETTE_ent, multi_layer_suggested_terms=multi_layer_suggested_terms)
#default_types_att = dbc.Row([dbc.Col(make_switch(label, colors_suggested_att[label],i)) for i, label in enumerate(SUGGESTED_TYPES_ATT)],
#            id="ex4cds-label-badge-group-att", )
#print(default_types_ent)
default_types_fact = dbc.Row([dbc.Col(make_switch(label, colors_suggested_fact[t],t, value=False)) for t, label in FACTUALITY.items()],
            id="ex4cds-label-badge-group-fact", )

default_types_temporal = dbc.Row([dbc.Col(make_switch(label, colors_suggested_temp[t],t, value=False)) for t, label in TEMPORAL.items()],
            id="ex4cds-label-badge-group-fact", )

default_types_risk = dbc.Row([dbc.Col(make_switch(label, colors_suggested_risk[t],t, value=False)) for t, label in RISK_FACTORS.items()],
            id="ex4cds-label-badge-group-risk", )

tooltips = {}
     

# Init_home_page_content(app_skeleton):

prediction_file_paths, classification_file_paths = get_result_cardio_files(RESULT_PATH_cardio)
text_ids = [k.split('.')[0] for k in prediction_file_paths.keys()]

select_document = html.Div(children = [html.H5('Document ID and Section'),
                                       dcc.Dropdown(options=[{'label':d, 'value':d} for d in text_ids], id='ex4cds-selected-id')],
)

# Row 2
section_checklist =  dbc.Checklist(id='ex4cds-cardio-section-checklist', options = [], value = [])  
   
# Row 3 data statistic
number_of_tokens = []
number_of_sentences = []
plot_data_statistic = html.Div(children = [html.H5('Number of sentences and words'), dcc.Graph(id="ex4cds-plot-number-of-sents-tokens")])
 
html_div_children = []
html_div_dropdown_ids = {}
left_container_children = [select_document,html.Br(), 
                        section_checklist,
                                 html.Hr(),
                                 plot_data_statistic
                                 ]
left_container =  dmc.Card(id='ex4cds-left-container',  
                           children=left_container_children,className = "two columns" , 
                           style = { "overflow":'scroll',
                            "background-color": "#f8f9fa", "height": "auto"}, withBorder=True,
                            shadow="sm",
                        radius="md",)


#container_exp_card = dmc.Card(children=[
#                        dmc.CardSection( 
#            inheritPadding=True,py="xs",), 
#                        dmc.SimpleGrid(cols=2, children=[container_exp, container_ann])], 
#                              shadow="sm", radius="sm",)

#container_vis = dmc.Card(id="ex4cds-container-vis", children=[],
#                         style={"height": "auto"},withBorder=True,shadow="sm", radius="md",)


container_main = html.Div(children= [semantic_type_row, 
                                     html.Hr(),
                                     dbc.Row([dbc.Col([container_exp], width=5), 
                                              dbc.Col([container_ann], width=5)]),
                                    html.Hr(),
                                    dbc.Row(children = [dbc.Col([dmc.Tabs(id='ex4cds-container-vis-tabs', 
                                                    orientation="horizontal", variant='pills',
                                                    persistence_type='local')], width=6),
                                                    dbc.Col([html.Frame(id='ex4cds-eval-plots')], width=4)])
                                    ])

store_tsv_sentences = dcc.Store(id='ex4cds-tsv-or-line-sentences-memory', storage_type='session')
store_section_tab = dcc.Store(id='ex4cds-section-tab-memory', storage_type = 'local')
store_section_sentence = dcc.Store(id='ex4cds-section-sentence-memory', storage_type='session')

store_prediction_files = dcc.Store(id="ex4cds-prediction-files-memory", storage_type="session", data=prediction_file_paths)
store_classification_files = dcc.Store(id="ex4cds-classification-files-memory", storage_type = "session", data=classification_file_paths)
store_prediction_dict = dcc.Store(id='ex4cds-prediction-memory', storage_type='local')
store_classification_dict = dcc.Store(id='ex4cds-classification-memory', storage_type='local')
store_vis_selected_types = dcc.Store(id='ex4cds-vis-selected-types-memory', storage_type='local')
store_spans_all_sentences = dcc.Store(id='ex4cds-spans-all-sentences-memory', storage_type='local')

title = html.Div(id='ex4cds-interface-title')
home_page_content = [store_prediction_files,
                    store_classification_files,
                    store_tsv_sentences,
                    store_spans_all_sentences,
                    store_section_tab,
                    store_vis_selected_types,
                            store_section_sentence, 
                            store_prediction_dict,
                            store_classification_dict,
                           left_container,
                        container_main
                                ]

layout = html.Div(children = home_page_content, id='ex4cds-home-page-content')

#print(text_ids)


@callback([Output('ex4cds-prediction-files-memory', 'data'), 
           Output('ex4cds-classification-files-memory', 'data'),
           Output('ex4cds-select-document-id', 'children')], 
          Input('ex4cds-select-domain', 'value'), prevent_initial_call = True)
def return_saved_results(domain):
    #CLINIC_DOMAIN = {'Cardiology':'Kardiologie', 'Nephrology':'Nephrologie'}
    dropdown_list = []
    prediction_file_paths = {}
    classification_file_paths = {}
    if domain:
        if domain == 'Kardiologie':
            prediction_file_paths, classification_file_paths = get_result_cardio_files(RESULT_PATH_cardio)
            text_ids = [k.split('.')[0] for k in prediction_file_paths.keys()]
            dropdown_list.append(dcc.Dropdown(options=[{'label':d, 'value':d} for d in text_ids], id='ex4cds-selected-id'))
        
        else:
            prediction_file_paths, classification_file_paths = get_result_ex4cds_files(RESULT_PATH_ex4cds)
            dropdown_list.append(dcc.Dropdown(options=[{'label':d, 'value':d} for d in prediction_file_paths.keys()], id='ex4cds-selected-id', multi=False))
    return prediction_file_paths, classification_file_paths, dropdown_list


@callback(Output('ex4cds-select-group-types', 'options'), 
          Input('ex4cds-select-groups', 'value'))
def return_label_types(group_name):
    options = []
    if group_name:
        options = [{'label': t+': '+ w, 'value': t} for t, w in multi_layer_suggested_terms[group_name].items()]
    return options

@callback([Output('ex4cds-tsv-or-line-sentences-memory', 'data'), 
           Output('ex4cds-cardio-section-checklist', 'options'),
           Output('ex4cds-cardio-section-checklist', 'value'),
            Output('ex4cds-prediction-memory', 'data'), 
           Output('ex4cds-classification-memory', 'data')], 
          [Input('ex4cds-selected-id', 'value'), 
           Input('ex4cds-prediction-files-memory', 'data'), 
           Input('ex4cds-classification-files-memory', 'data')])
def return_cardio_tsv_sentences(text_id, prediction_file_paths, classification_file_paths):
    section_options = []
    sentences_dict = {}
    prediction_dict = {}
    classification_dict = {}
    
    if text_id and '_' not in text_id:
        prediction_dict, classification_dict, _ = return_json_cardio_dict(text_id, RESULT_PATH_cardio, prediction_file_paths=prediction_file_paths, classification_file_paths=classification_file_paths)
        section_options = make_section_options(prediction_dict.keys())
           
        tsv_lines = get_tsv_lines(text_id)
        sentences_dict = retrieve_sentences(tsv_lines)
    
    return sentences_dict, section_options, [], prediction_dict, classification_dict
        

@callback([Output('ex4cds-container-vis-tabs', 'children'),
           Output('ex4cds-container-vis-tabs', 'value'), 
           Output('ex4cds-section-tab-memory', 'data'),
           Output('ex4cds-spans-all-sentences-memory', 'data')],
        [
         Input('ex4cds-selected-id', 'value'),
         Input('ex4cds-cardio-section-checklist', 'value'), 
         Input('ex4cds-vis-selected-types-memory', 'data'),
         Input('ex4cds-tsv-or-line-sentences-memory', 'data'),
         Input('ex4cds-prediction-memory', 'data'), 
         Input('ex4cds-classification-memory', 'data')], prevent_initial_call=True)
def make_section_tabs(text_id, sections, selected_types,sentences_dict, prediction_current, classification_current):
        sec_sentences = {}
        section_tab = {}
        tabs = []
        labels = []
        dmc_tabs = []
        spans_all_sentences = []
        if sections is not None:
            for i, sec in enumerate(sections):
                sentence_ids, scores_dict_sec, classification_dict_sec = get_prediction_classification_section(prediction_current, classification_current, sec_tab=sec)
                spans_all_sentences = get_spans_all_sentences(sentence_ids, sentences_dict)
                spans_all_sentences = get_sentence_classification(classification_dict_sec, spans_all_sentences, selected_sem_types=selected_types, threshold='0.5')
                labels.append(dmc.Tab(sec, value='tab-{}'.format(i)))
                lines_str_html_ths = ''
                for t, sent_spans in spans_all_sentences.items():
                
                    lines_str_html = format_span_line_markup(text=sent_spans['text'], spans=sent_spans['ner_spans'], palette=PALETTE_types)
                    lines_str_html_ths += '<hr>{}</hr>'.format(''.join(lines_str_html))
                    
                one_tab = make_one_section_ner_tab('tab-{}'.format(i), lines_str_html_ths)
                tabs.append(one_tab)
                section_tab['tab-{}'.format(i)] = sec
                
    
                
        dmc_tabs = [dmc.TabsList(labels)] + tabs
        return dmc_tabs, 'tab-0', section_tab, spans_all_sentences
        

@callback(Output('ex4cds-select-types-explain', 'children'), [
                                                       Input('ex4cds-prediction-memory', 'data'),
                                                       Input('ex4cds-container-vis-tabs', 'value'),
                                                       Input('ex4cds-section-tab-memory', 'data')])
def make_explain_types_options( prediction_dict,tab, sec_tab_memory):
    
    if  sec_tab_memory is not None and tab in sec_tab_memory:
        #print(tab)
        sec_current = sec_tab_memory[tab]
        #print(sec_current)
        
        prediction_dict = prediction_dict[sec_current]
            
        select_options = make_select_types_to_explain(prediction_dict)
    
        return [select_options]
    else:
        return html.Div([
                      dcc.Dropdown(id='ex4cds-types-options')
                      ])

@callback([Output('ex4cds-container-exp', 'children')],
                [Input('ex4cds-prediction-memory', 'data'), 
                 Input('ex4cds-tsv-or-line-sentences-memory', 'data'),
                 Input('ex4cds-spans-all-sentences-memory', 'data'),
                 Input('ex4cds-types-options', 'value'), 
                 Input('ex4cds-container-vis-tabs', 'value'),
                 Input('ex4cds-section-tab-memory', 'data')],
                 prevent_initial_call=True)
def make_sentence_explanation( scores_dict, tsv_sentences, spans_all_sentences, sem_types, tab, sec_tab_memory):
        
        main_children = [] 

        header = dbc.Row(children=[dbc.Col(dmc.Text('Type:', color='blue', size='l', 
                                                 weight=500,), width=3), dbc.Col(dmc.Text('Sentences:', color='blue', weight=500), width=6)])
        main_children.append(header)
        
        word_scores = {}
        
        if sem_types is not None and tab in sec_tab_memory:
                sec_tab = sec_tab_memory[tab]
            
                word_scores = get_sentence_scores_by_type_cardio(scores_dict, tsv_sentences, sec_tab, sem_types)
                #print(word_scores)
                #main_children = [html.P('Here to show the heatmaps for type {}'.format(type))]
      
        if len(word_scores) > 0 :
            for ty, sent_scores in word_scores.items():
                spoiler_children = []
                for current_sent_score in sent_scores:
                    spoiler_children.append(html.Hr())
                    type_p = dash_html_prediction_score(current_sent_score['type_scores'][0],current_sent_score['type_scores'][1], cmap_name='bwr' )
                    word_p = dash_html_prediction_score(current_sent_score['word_scores'][0],current_sent_score['word_scores'][1], cmap_name='bwr' )
                    spoiler_children.append(dbc.Row(children=[dbc.Col(dmc.Text(type_p),width=3),dbc.Col(dmc.Text(word_p, style={'overflow-x': 'auto',}),width=9)]))
                main_children.append(dmc.Spoiler(showLabel="Show more",hideLabel="Hide",maxHeight=200, children = spoiler_children))
        #print(len(main_children))
        return [dmc.Stack(children=main_children)]
    
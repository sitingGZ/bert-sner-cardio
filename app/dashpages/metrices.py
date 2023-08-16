import os
import json
import pandas as pd
import plotly
from sklearn.metrics import classification_report
from dashpages.utils import get_result_cardio_files, return_json_cardio_dict
import collections

from dash import html

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


THRESHOLD = '0.5' 

IDEAL_SECTIONS_IDS = {'Befunde': ['118', '122', '174', '190', '197', '198', '218', '225', '235', '247', '263', '275', '288'], 
                      'Diagnosen': ['1', '13', '106', '109', '119', '132', '133', '150', '164', '165', '172'], 
                      'Zusammenfassung': ['21', '101', '135', '149', '151', '209', '210', '219', '249', '251', '256', '259', '262', '264', '296', '311', '322']}


def get_classification_results(path, prediction_file_paths, classification_file_paths,  text_id, section):   
    prediction_dict, classification_dict_original = return_json_cardio_dict(text_id, path, prediction_file_paths=prediction_file_paths, classification_file_paths=classification_file_paths)
    #section_options, section_values = make_section_options(list(prediction_dict.keys()))
    #section_radios_options = [{'label' : s, 'value' :s} for s in prediction_dict.keys()]
    classification_dict = {}
    #for section, section_classification in classification_dict_original.items():
    if section in classification_dict_original: 
        section_classification = classification_dict_original[section]
        #classification_dict[section] = {}
        for group in section_classification.keys():
            if group == "Zeitliche Information":
                classification_dict['Temporal'] = section_classification[group][THRESHOLD]
            else:
                classification_dict[group] = section_classification[group][THRESHOLD]
                
    return classification_dict

def get_section_sentences_dict(path, prediction_file_paths, classification_file_paths):
    section_sentences = {}
    for section, text_ids in IDEAL_SECTIONS_IDS.items():
        section_sentences[section] = {'Sentences': [], 'Group Annotations': {group: [] for group in multi_layer_suggested_terms.keys()}}
        for text_id in text_ids:
            cls_dict = get_classification_results(path, prediction_file_paths, classification_file_paths, text_id, section)
            #print(text_id)
            for group, sentences in cls_dict.items():
            #print(sentences)
            #section_sentences[section]['Group Annotations'][group] = []
                for sentence in sentences:
                    section_sentences[section]['Group Annotations'][group] += [sent[2] for sent in sentence if sent[2] != 'O']
            
            section_sentences[section]['Sentences'].append(len(list(cls_dict.values())[0]))
            
   
    return section_sentences

def get_amount_sentences(section_sentences, classification_annotator, annotator):
    amount_sentences = []
    for section in section_sentences:
        amount_sentences.append({'Section': section, 'Sentence Amount': sum(section_sentences[section]['Sentences']), 'Annotator' :'BERT-SNER'})
        if section in classification_annotator:
            section_amount = []
            for text_id, sentences in classification_annotator[section].items():
                #print(sentences)
                section_amount.append(len(list(sentences.values())[0]))
            amount_sentences.append({'Section': section, 'Sentence Amount': sum(section_amount), 'Annotator' :annotator})
        else:
            amount_sentences.append({'Section': section, 'Sentence Amount': 0, 'Annotator' :annotator})

    return amount_sentences

def get_amount_annotations(section_sentences, classification_annotator, annotator):
    amount_annotation = []
    for section in section_sentences:
        for group, annotations in section_sentences[section]['Group Annotations'].items():
            #print(annotations)
            annotation_dict = collections.Counter(annotations) if len(annotations) > 0 else {}
        #   print("prediction" )
        #   print(annotation_dict)
            for term, words in multi_layer_suggested_terms[group].items():
                if term in annotation_dict:
                    amount_annotation.append({'Section': section, 'Group': group, 'Term': term + ': ' + words, 'Amount': annotation_dict[term], 'Annotator': 'BERT-SNER'})
                else:
                     amount_annotation.append({'Section': section, 'Group': group, 'Term': term + ': ' + words, 'Amount': 0, 'Annotator': 'BERT-SNER'})
        #print("******")               
        if section in classification_annotator:
            for text_id, sentences in classification_annotator[section].items():
                for group, prediction in sentences.items():
                    #print(group, len(prediction))
                    annotation_ann = []
                    for sentence in prediction:
                        group_prediction = [pred[2] for pred in sentence if pred[2] != 'O']
                        annotation_ann += group_prediction
                   
                    annotation_dict_ann = collections.Counter(annotation_ann) if len(annotation_ann) > 0 else {}
                    #print("annotation annotator")
                    #print(annotation_dict_ann)
                    for term, words in multi_layer_suggested_terms[group].items():
                        if term in annotation_dict_ann:
                               amount_annotation.append({'Section': section, 'Group': group, 'Term': term + ': ' + words, 'Amount': annotation_dict_ann[term], 'Annotator': annotator})
                        else:
                             amount_annotation.append({'Section': section, 'Group': group, 'Term': term + ': ' + words, 'Amount': 0, 'Annotator': annotator})

    return amount_annotation

            
def get_y_pred_true(classification_annotator, path, prediction_file_paths, classification_file_paths):
    predictions = {group: [] for group in multi_layer_suggested_terms.keys()}
    annotations_ref = {group: [] for group in multi_layer_suggested_terms.keys()}

    for section in classification_annotator:
        for text_id, sentences in classification_annotator[section].items():
            prediction_current = get_classification_results(path, prediction_file_paths, classification_file_paths, text_id, section)
            #print(prediction_current.keys())
            for group, annotations in sentences.items():
                #print(group, len(annotations))
               
                for sentence in annotations:
                    group_anno = [pred[2] for pred in sentence]
                    annotations_ref[group].extend(group_anno)
                for sentence_pred in prediction_current[group]:
                    group_prediction = [pred[2] for pred in sentence_pred]
                    predictions[group].extend(group_prediction)
    return predictions, annotations_ref 

def get_group_classification_report(predictions, annotations_ref):
    group_results = {}
    for group, group_prediction in predictions.items():
        if len(group_prediction) >0:
            classification_result = classification_report(y_pred = group_prediction, y_true = annotations_ref[group])
            group_results[group] = classification_result.split('\n\n')
    return group_results 

def make_report_table(group_report):
    
    heads = [html.Th('label')] + [html.Th(l) for l in group_report[0].split()]
    header = [html.Thead(html.Tr(heads) )]
    
    rows = []
    for line in group_report[1].split('\n'):
        row = html.Tr([html.Td(t) for t in line.split()])
        rows.append(row)
    
    rows.append(html.Tr([html.Td() for _ in range(5)]))
    for i, line in enumerate(group_report[-1].split('\n')):
        line_split = line.split()
        if i == 0:
            accurarcy_row = html.Tr([html.Td(line_split[0]), html.Td(), html.Td(), html.Td(line_split[-2]), html.Td(line_split[-1])])
            rows.append(accurarcy_row)
        else:
            if len(line_split)> 5:
                score_row = html.Tr([html.Td(' '.join(line_split[:2]))] + [html.Td(s) for s in line_split[2:]])
                rows.append(score_row)
    
    body = [html.Tbody(rows)]
    return header, body
                
        
                    
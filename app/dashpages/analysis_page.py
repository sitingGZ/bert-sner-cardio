import copy
import dash
from dash import dash_table

from dash import no_update
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc

from dash import dcc, html, Dash
from dash import MATCH, ALL
from dash.dependencies import Output, Input, State
from flask import Flask
import pathlib
import pandas as pd
import json 
import os
import logging
import matplotlib.pyplot as plt

from modules.BERT2span_semantic_disam import BERT2span
from modules.helpers import CustomThread, get_host_name
from modules.inference import final_label_results_rescaled
from modules.helpers import load_config, set_seed

from copy import deepcopy
import torch
from transformers import  AutoTokenizer, AutoModelForMaskedLM



ANALYSISPATH = {'span': 'Explanation', 'href':'/explanation'}
multi_layer_suggested_terms = {'Physical Object': 
                               {'T017': ('Anatomie', 
                                         "Ein normaler oder pathologischer Teil der Anatomie oder strukturellen Organisation eines Organismus." ),
                                "T200": ("Arzneimittel", 
                                         "Ein pharmazeutisches Praeparat, wie es vom Hersteller produziert wird.  Der Name umfasst in der Regel die Substanz, ihre Staerke und die Form, kann aber auch die Substanz und nur einen der beiden anderen Punkte enthalten."),
                                "T103": ("Chemikalien",
                                 "Verbindungen oder Stoffe mit einer bestimmten molekularen Zusammensetzung.  Chemikalien werden im Netz aus zwei verschiedenen Perspektiven betrachtet, der funktionalen und der strukturellen.  Fast jedem chemischen Konzept sind mindestens zwei Typen zugeordnet, im Allgemeinen einer aus der Strukturhierarchie und mindestens einer aus der Funktionshierarchie."),
                             "T074": ("Medizinisches Geraet", 
                                      "Ein hergestellter Gegenstand, der in erster Linie zur Diagnose, Behandlung oder Vorbeugung von physiologischen oder anatomischen Stoerungen verwendet wird.")
                               }, 
                               " Conceptual Entity": {
                                   'T201': ("Klinisches Attribut", 
                                            "Eine beobachtbare oder messbare Eigenschaft oder ein Zustand eines Organismus von klinischem Interesse."),
                                   'T081': ("Quantitatives Konzept",
                                             "Ein Konzept, das die Dimensionen, die Quantitaet oder die Kapazitaet von etwas unter Verwendung einer Masseinheit betrifft oder den quantitativen Vergleich von Entitaeten beinhaltet."),
                                   'T184': ("Zeichen oder Symptom", 
                                            "Eine beobachtbare Manifestation einer Krankheit oder eines Zustands, die auf einer klinischen Beurteilung beruht, oder eine Manifestation einer Krankheit oder eines Zustands, die vom Patienten erlebt und als subjektive Beobachtung berichtet wird."),
                                   'T034': ("Labor- oder Testergebnis", 
                                            "Das Ergebnis eines spezifischen Tests zur Messung eines Attributs oder zur Bestimmung des Vorhandenseins, des Nichtvorhandenseins oder des Grades eines Zustands."),
                                   'T079': ("Zeitliches Konzept", 
                                           "Ein Konzept, das sich auf die Zeit oder die Dauer bezieht." )},
                               "Activity" : {
                                  'T059': ("Laborverfahren", 
                                           "Ein Verfahren, eine Methode oder eine Technik zur Bestimmung der Zusammensetzung, Menge oder Konzentration einer Probe, die in einem klinischen Labor durchgefuehrt wird.  Hierunter fallen auch Verfahren zur Messung von Reaktionszeiten und -raten."),
                                   'T060': ("Diagnostisches Verfahren",
                                            "Ein Verfahren, eine Methode oder eine Technik, die dazu dient, die Art oder Identitaet einer Krankheit oder Stoerung zu bestimmen.  Dies schliesst Verfahren aus, die hauptsaechlich an Proben in einem Labor durchgefuehrt werden."), 
                                   'T061': ("Therapeutisches oder praeventives Verfahren",
                                            "Ein Verfahren, eine Methode oder eine Technik, die dazu bestimmt ist, einer Krankheit oder Stoerung vorzubeugen oder die koerperliche Funktion zu verbessern, oder die im Rahmen der Behandlung einer Krankheit oder Verletzung eingesetzt wird.")}, 
                               "Phenomenon or Process":{
                                'T037': ("Verletzung oder Vergiftung", 
                                         "Eine traumatische Wunde, Verletzung oder Vergiftung, die durch eine aeussere Einwirkung oder Kraft verursacht wurde."),
                               'T039': ("Physiologische Funktion", 
                                       "Ein normaler Prozess, eine normale Aktivitaet oder ein normaler Zustand des Koerpers." ),
                               'T046': ("Pathologische Funktion", 
                                         "Ein gestoerter Prozess, eine gestoerte Aktivitaet oder ein gestoerter Zustand des Organismus als Ganzes, eines oder mehrerer Koerpersysteme oder mehrerer Organe oder Gewebe.  Dazu gehoeren normale Reaktionen auf einen negativen Reiz sowie pathologische Zustaende oder Zustaende, die weniger spezifisch sind als eine Krankheit.  Pathologische Funktionen haben haeufig systemische Auswirkungen."),
                               'T047': ("Krankheit",
                                        "Ein Zustand, der einen normalen Prozess, Zustand oder eine normale Aktivitaet eines Organismus veraendert oder stoert.  Er ist in der Regel durch die abnorme Funktion eines oder mehrerer Systeme, Teile oder Organe des Wirts gekennzeichnet.  Hierunter faellt auch ein Komplex von Symptomen, die eine Stoerung beschreiben."), 
                               'T048': ("Psychische oder Verhaltensstoerung", 
                                        "Eine klinisch bedeutsame Funktionsstoerung, deren Hauptmanifestation verhaltensbezogen oder psychologisch ist.  Diese Stoerungen koennen identifizierte oder vermutete biologische aetiologien oder Manifestationen haben.")},
                               "Health State":{'T300': ("Gesunder Zustand", ),  
                                               "T301": ("Verschlechter Zustand", )},
                               "Factuality": {'T400': ('kein oder negiert', ), 
                                              'T401' : ('gering', ),  
                                              'T402' : ('fraglich',),
                                              'T403': ('zukueftig', ),
                                              'T404': ('unwahrscheinlich', )},
                               "Temporal": {'T500': ('aktuelles Ereignis', ),
                                            'T501': ('Vergangenheit zum aktuellen Ereignis',),
                                            'T502': ('vergangenes Ereignis', ),
                                            'T503': ('zukuenftiges Ereignis', )}}


TAB_VIEWS = ['Applied Datasets', 'Model Performance', 'Feature Importance', 'Error Analysis' , 'What-if']

cards = dmc.SimpleGrid(id='analysis-cards', cols=3, children = [
    dbc.Card(
    children = [
        html.H5(label), html.Div(id = 'card-{}-{}'.format(i, label), style={'height':'300px'})]) for i, label in enumerate(TAB_VIEWS)])

left_sidebar = html.Div(children = [dmc.Card(id='analysis-sidebar',  
                        children=[html.H5('Configuration'), dbc.Checklist(options=[])],
                        style = { "overflow":'scroll',"background-color": "#f8f9fa",}, withBorder=True,
                            shadow="sm",
                        radius="md")],  className = "two columns")


right_container = html.Div(id="analysis-container", children=[cards], className = "ten columns",)
    

layout = html.Div(children = [left_sidebar, right_container])



# Tab applied datasets

# Tab model performance

# Feature importance

# what-if 

# Error analysis



#

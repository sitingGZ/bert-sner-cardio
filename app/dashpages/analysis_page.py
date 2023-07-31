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

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

from dashpages import home_page, analysis_page, annotate_page_cardio

from copy import deepcopy

from transformers import  AutoTokenizer, AutoModelForMaskedLM

#app =  Dash(__name__, requests_pathname_prefix = '/',meta_tags=[{ "content": "width=device-width, initial-scale=1"}], 
#           external_stylesheets=[dbc.themes.CERULEAN, dbc.icons.FONT_AWESOME], suppress_callback_exceptions=True)

app = Dash(__name__, requests_pathname_prefix = '/', external_stylesheets=[dbc.themes.CERULEAN, dbc.icons.FONT_AWESOME], suppress_callback_exceptions=True)

app.title= "Clinical Text Semantic Analytics with BERT-SNER"
server = app.server
app.config.suppress_callback_exceptions = True

dfki_colors = ["#2980b9", "#3498db"]

def make_navbar(nav_path_list, nav_path_dict = None):
    NAV_STYLE = {
    "padding":"10px 20px",
    "font-size": "35px",
    "font":'bold',
    'color': 'dark'}
    
    brand = dbc.NavItem(dbc.NavLink("Overview", active=True, href=home_page.HOMEPATH['href']), style={"font-size": "30px",  'font-color':dfki_colors[0], 'font':'bold', 'text-align':'left'})
    
    nav_links = [dbc.Col(brand)] + [dbc.Col(dbc.NavItem(dbc.NavLink(path['span'], active=True, href=path['href']), style={"font-size": "25px"})) for path in nav_path_list]
    if nav_path_dict is not None:
        nav_dropdown_menus = []
        for k, paths in nav_path_dict.items():
            current_dropdown_menu = dbc.DropdownMenu(children=[dbc.DropdownMenuItem(path['span'], href=path['href'], style={"font-size": "25px",}) for path in paths], 
            nav=True, in_navbar=True, label=k, style={"font-size": "25px", "font-color":dfki_colors[1]})
            nav_dropdown_menus.append(current_dropdown_menu)
    
        nav_links.extend([dbc.Col(nav_dropdown_menus)])        
    
    return dbc.Navbar(children= dbc.Row(nav_links), light=True)
    

nav_path_list = [analysis_page.ANALYSISPATH]
nav_path_dict = {'Annotation Projects': [annotate_page_cardio.ANNOTATEPATH]}
navbar = make_navbar(nav_path_list=nav_path_list, nav_path_dict=nav_path_dict)
page_content = html.Div(id='page-toggle', children= [])
app.layout = html.Div([dcc.Location(id = 'url'),navbar] + [html.Hr(), page_content])


@app.callback(Output('page-toggle', 'children'), Input('url', 'pathname'),prevent_initial_call=True)
def render_page_content(path):
            if path == home_page.HOMEPATH['href']:
                page = home_page.layout
            elif path == annotate_page_cardio.ANNOTATEPATH['href']:
                page = annotate_page_cardio.layout
            elif path == analysis_page.ANALYSISPATH['href']:
                page = analysis_page.layout
                #    page = extraction.render_extraction_page(app)
                #elif path == data.Path['href']:
                #    page = data.render_data_page(app)
        #elif path == diag2note.Path['href']:
        #    page = diag2note.render_diag2note_page(app)
            else:
                page = home_page.layout
            return page
        
   
        
if __name__ == "__main__":
    
#    import sys
    #config_path = 'configs/transformer_config.yaml'
    #main(config_path)
    
    app.run_server(debug=True)
    
    
    #instance.kill()
    
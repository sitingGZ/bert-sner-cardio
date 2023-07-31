import re
import ast
from collections import defaultdict
from textwrap import TextWrapper
from html import escape
from intervaltree import IntervalTree as Intervals
from ipymarkup.show import show_html
from ipymarkup.record import Record
from ipymarkup.palette import Color, MaterialRgb, material, Palette
from ipymarkup.span import order_spans, span_text_sections, format_span_box_markup, format_span_line_markup, wrap_multilines, prepare_spans, get_multilines
from ipymarkup.dep import format_dep_markup
import matplotlib.pyplot as plt

from xml.etree import ElementTree

from dash import html, dcc
import dash_mantine_components as dmc

AMBER = Color(
    'amber',
    background=material('Amber', '50'),
    border=material('Amber', '300'),
    text=material('Amber', '400'),
    line=material('Amber', '400')
)

BLUE = Color(
    'blue',
    background=material('Blue', '50'),
    border=material('Blue', '300'),
    text=material('Blue', '400'),
    line=material('Blue', '400')
)
GREEN = Color(
    'green',
    background=material('Green', '50'),
    border=material('Green', '300'),
    text=material('Green', '400'),
    line=material('Green', '400')
)
RED = Color(
    'red',
    background=material('Red', '50'),
    border=material('Red', '300'),
    text=material('Red', '400'),
    line=material('Red', '400')
)

PINK = Color(
    'pink',
    background=material('Pink', '50'),
    border=material('Pink', '300'),
    text=material('Pink', '400'),
    line=material('Pink', '400')
)
ORANGE = Color(
    'orange',
    background=material('Orange', '50'),
    border=material('Orange', '300'),
    text=material('Orange', '400'),
    line=material('Orange', '400')
)
PURPLE = Color(
    'purple',
    background=material('Deep Purple', '50'),
    border=material('Deep Purple', '300'),
    text=material('Deep Purple', '400'),
    line=material('Deep Purple', '400')
)
BROWN = Color(
    'brown',
    background=material('Brown', '50'),
    border=material('Brown', '300'),
    text=material('Brown', '400'),
    line=material('Brown', '400')
)
GREY = Color(
    'grey',
    text=material('Grey', '500'),
    line=material('Grey', '400')
)



CYAN = Color(
    'cyan',
    background=material('Cyan', '50'),
    border=material('Cyan', '300'),
    text=material('Cyan', '400'),
    line=material('Cyan', '400')
)

LIME = Color(
    'lime',
    background=material('Lime', '50'),
    border=material('Lime', '300'),
    text=material('Lime', '400'),
    line=material('Lime', '400')
)

TEAL = Color(
    'teal',
    background=material('Teal', '50'),
    border=material('Teal', '300'),
    text=material('Teal', '400'),
    line=material('Teal', '400')
)

Indigo = Color(
    'Indigo',
    background=material('Indigo', '50'),
    border=material('Indigo', '300'),
    text=material('Indigo', '400'),
    line=material('Indigo', '400')
)




colors_ent = [ORANGE, RED, TEAL, Indigo ,BROWN, GREEN,  LIME]

color_fact = [GREEN]

color_temporal = [BLUE]

color_risk_att = [GREEN, RED, GREY]

ents = ['LabValues', 'DiagLab', 'Measure', 'Condition', 'Medication', 'Process', ',HealthState', 'TimeInfo', 'Other']
temporal = ['past', 'past_present', 'present', 'future']
factuality = ['negative', 'speculated', 'minor', 'unlikely', 'possible_future']
risk_att = ['increase_risk_factor', 'increase_symptom', 'decrease_risk_factor', 'decrease_symptom']
rels = ['has_TimeInfo', 'has_Measure', 'has_State']

def retrieve_annos(file):
    annotate_lines = [l.strip().split('\t') for l in open(file)]
    spans_ent = {l[0]: l[1].split() for l in annotate_lines if l[0][0] == 'T' }
    spans_att = {l[0]: l[1].split()  for l in annotate_lines if l[0][0] == 'A'}
    spans_rl = {l[0]:l[1].split() for l in annotate_lines  if l[0][0] == 'R'}
    return spans_ent, spans_att, spans_rl

def retrieve_ents(spans_ent, spans_att, to_show=[], att='Factuality'):
    # box + line show
    # att is either 'Factuality' or 'Risk'
    if len(to_show) == 0: 
        to_show = ['LabValues', 'DiagLab', 'Measure', 'Condition', 'Medication', 'Process', ',HealthState', 'TimeInfo', 'Other']
    
    spans_to_show = [(int(v[1]), int(v[2]), k) for k,v in spans_ent.items() if v[0] in to_show]
    ent_dict = {k:v[0] for k, v in spans_ent.items()}
    att_dict = {v[1]:v[2] for k, v in spans_att.items() if v[0] == att} 
    return spans_to_show, ent_dict, att_dict

def retrieve_conl(spans_ent):
    # line show
    conclusions = [(int(v[1]), int(v[2]), 'Conclusion') for k,v in spans_ent.items() if v[0] == 'Conclusion' ]
    return conclusions

import numpy as np
def find_most_closest(idx, indices):
    min_d = np.array([abs(d-idx) for d in indices])
    min_i = indices[np.argmin(min_d)]
    return min_i

    
def resolve_rel_idx(spans_rl, spans_ent, idx_w):
    rl_tuples = []
    for k, v in spans_rl.items():
        left = v[1].split(':')[1]
        s_left = int(spans_ent[left][1])
        if s_left not in idx_w:
            s_left = find_most_closest(s_left, list(idx_w.keys()))
        w_left = idx_w[s_left][0]
        right = v[2].split(':')[1]
        s_right = int(spans_ent[right][1])
        if s_right not in idx_w:
            s_right = find_most_closest(s_right, list(idx_w.keys()))
        w_right = idx_w[s_right][0]
        if w_left == w_right:
            if left > right:
                w_right += 1
            else:
                w_right -=1
        rl_tuples.append((w_left, w_right, v[0]))
    return rl_tuples



def retrieve_temp(spans_ent, spans_att, to_show=[], att='Temporal_Element'):
    # box + line show
    spans_to_show = [(int(v[1]), int(v[2]), k) for k,v in spans_ent.items() if v[0] == 'Temporal']
    ent_dict = {k:v[0] for k, v in spans_ent.items()}
    att_dict = {v[1]:v[2] for k, v in spans_att.items() if v[0] == att} 
    return spans_to_show, ent_dict, att_dict

import numpy as np
def find_most_closest(idx, indices):
    min_d = np.array([abs(d-idx) for d in indices])
    min_i = indices[np.argmin(min_d)]
    return min_i

    
def resolve_rel_idx(spans_rl, spans_ent, idx_w):
    rl_tuples = []
    for k, v in spans_rl.items():
        left = v[1].split(':')[1]
        s_left = int(spans_ent[left][1])
        if s_left not in idx_w:
            s_left = find_most_closest(s_left, list(idx_w.keys()))
        w_left = idx_w[s_left][0]
        right = v[2].split(':')[1]
        s_right = int(spans_ent[right][1])
        if s_right not in idx_w:
            s_right = find_most_closest(s_right, list(idx_w.keys()))
        w_right = idx_w[s_right][0]
        if w_left == w_right:
            if left > right:
                w_right += 1
            else:
                w_right -=1
        rl_tuples.append((w_left, w_right, v[0]))
    return rl_tuples

def format_span_box_line_markup(text, spans, entities=None, attributes=None, palette_ent=None, palette_att = None,
                            width=200, line_gap=16, line_width=3,
                            label_size=14, background='white'):
    spans = order_spans(prepare_spans(spans))
    multilines = list(get_multilines(spans))

    level_width = line_gap + line_width
    yield (
        '<div class="tex2jax_ignore" style="'
        'white-space: pre-wrap'
        '">'
    )
    for offset, line, multilines in wrap_multilines(text, multilines, width):
        yield '<div>'  # line block
        for text, multi in span_text_sections(line, multilines):
            text = escape(text)
            if not multi:
                yield (
                    '<span style="display: inline-block; '
                    'vertical-align: top">'
                )
                yield text
                yield '</span>'
                continue

            level = max(_.level for _ in multi.lines)
            margin = (level + 1) * level_width
            yield (
                '<span style="display: inline-block; '
                'vertical-align: top; position: relative; '
                'margin-bottom: {margin}px">'.format(
                    margin=margin
                )
            )

            for line in multi.lines:
                padding = line_gap + line.level * level_width
                color_ent = palette_ent.get(entities[line.type])
                if attributes is not None and line.type in attributes:
                    color_att = palette_att.get(attributes[line.type])
 
                    yield (
                        '<span style="'
                        'padding: 1px; '
                        'border-radius: 2px; '
                        'border: 0.5px solid {border}; '
                        'background: {background}; '
                        'border-bottom: {line_width}px solid {color}; '
                        'padding-bottom: {padding}px'
                        '">'.format(
                            background=color_ent.background.value,
                            border=color_ent.border.value,
                            line_width=line_width,
                            padding=padding,
                            color=color_att.line.value
                            
                        
                        )
                    )
                else:
                    #level = max(_.level for _ in multi.lines)
                    margin = (level + 1) * level_width
                    yield (
                        '<span style="display: inline-block; '
                        'vertical-align: top; position: relative; '
                        'margin-bottom: {margin}px; '
                        'padding: 1px; '
                        'border-radius: 2px; '
                        'border: 1px solid {border}; '
                        'background: {background}'
                        '">'.format(
                            margin=margin,
                            background=color_ent.background.value,
                            border=color_ent.border.value
                        
                        )
                    )
                yield text
                yield '</span>'

            for line in multi.lines:
                if not line.type or offset + multi.start != line.start:
                    continue

                bottom = -line.level * level_width - line_gap
                color_ent = palette_ent.get(entities[line.type])
                
                yield (
                '<span style="'
                'vertical-align: middle; '
                'margin-left: 2px; '
                'font-size: 1em; '
                'color: {color};'
                '">'.format(
                    color=color_ent.text.value
                    )
                )
                yield entities[line.type]
                yield '</span>'
                if line.type in attributes:
                    yield (
                        '<span style="'
                        'font-size: {label_size}px; line-height: 1; '
                        'white-space: nowrap; '
                        'text-shadow: 1px 1px 0px {background}; '
                        'position: absolute; left: 0; '
                        'bottom: {bottom}px">'.format(
                            label_size=label_size,
                            background=background,
                            bottom=bottom
                        )
                    )
                    yield attributes[line.type]
                else:
                    margin = (level + 1) * level_width
                    yield (
                        '<span style="display: inline-block; '
                        'vertical-align: top; position: relative; '
                        'margin-bottom: {margin}px; '
                        '">'.format(
                            margin=margin
                        
                        )
                    )
                    
                yield '</span>'
                
                

            yield '</span>'  # close relative
        yield '</div>'  # close line
    yield '</div>'


#box_line_lines = format_span_box_line_markup(text=text_inj, spans=spans, entities=ent_dict, attributes=att_dict, palette_ent=PALETTE_ent, palette_att=PALETTE_att)
def get_w_idx(text_split):
    idx_w = {0:0}
    #text_list = text_inj[0].split()
    past = 0
    for i, w in enumerate(text_split):

        if i> 0:
            past += len(text_split[i-1])+1
            idx_w[past] = (i, w)
        else:
            idx_w[0] = (0, w)
    return idx_w

def convert_html_to_dash(html_code,  dash_modules = [html, dcc]):
    """Convert standard html (as string) to Dash components.

    Looks into the list of dash_modules to find the right component (default to [html, dcc, dbc])."""
    
    def find_component(name):
        for module in dash_modules:
            try:
                return getattr(module, name)
            except AttributeError:
                pass
        raise AttributeError(f"Could not find a dash widget for '{name}'")

    def parse_css(css):
        """Convert a style in ccs format to dictionary accepted by Dash"""
        return {k: v for style in css.strip(";").split(";") for k, v in [style.split(":")]}

    def parse_value(v):
        try:
            return ast.literal_eval(v)
        except (SyntaxError, ValueError):
            return v

    parsers = {"style": parse_css, "id": lambda x: x}

    def _convert(elem):
        comp = find_component(elem.tag.capitalize())
        children = [_convert(child) for child in elem]
        if not children:
            children = elem.text
        attribs = elem.attrib.copy()
        if "class" in attribs:
            attribs["className"] = attribs.pop("class")
        attribs = {k: parsers.get(k, parse_value)(v) for k, v in attribs.items()}

        return comp(children=children, **attribs)

    et = ElementTree.fromstring(html_code)

    return _convert(et)


# Word relevance heatmap
def rescale_score_by_abs(score, max_score, min_score):
 """
 Normalize the relevance value (=score), accordingly to the extremal relevance values (max_score and min_score),
 for visualization with a diverging colormap.
 i.e. rescale positive relevance to the range [0.5, 1.0], and negative relevance to the range [0.0, 0.5],
 using the highest absolute relevance for linear interpolation.
 """

 # CASE 1: positive AND negative scores occur --------------------
 if max_score > 0 and min_score < 0:

  if max_score >= abs(min_score):  # deepest color is positive
   if score >= 0:
    return 0.5 + 0.5 * (score / max_score)
   else:
    return 0.5 - 0.5 * (abs(score) / max_score)

  else:  # deepest color is negative
   if score >= 0:
    return 0.5 + 0.5 * (score / abs(min_score))
   else:
    return 0.5 - 0.5 * (score / min_score)

    # CASE 2: ONLY positive scores occur -----------------------------
 elif max_score > 0 and min_score >= 0:
  if max_score == min_score:
   return 1.0
  else:
   return 0.5 + 0.5 * (score / max_score)

 # CASE 3: ONLY negative scores occur -----------------------------
 elif max_score <= 0 and min_score < 0:
  if max_score == min_score:
   return 0.0
  else:
   return 0.5 - 0.5 * (score / min_score)


def getRGB(c_tuple):
 return "#%02x%02x%02x" % (int(c_tuple[0] * 255), int(c_tuple[1] * 255), int(c_tuple[2] * 255))

def dash_span_word(word, score, colormap):
    if score == None:
        #score = 0.5
        score = 0.0
    word = word.strip('\n')
    style = {'margin-left': '1px', 'margin-right': '2px', 'background-color': getRGB(colormap(score))}
    span = html.Span([word], style = style)
    return span

def html_span_word(word, score, colormap):
    if score == None:
        #score = 0.5
        score = 0.0
    word = word.strip('\n')
    style = {'margin-left': '1px', 'margin-right': '2px', 'background-color': getRGB(colormap(score))}
    span = html.Span([word], style = style)
    return span
    
def html_span_word_check(word, score, colormap):
    if score == None:
        #score = 0.5
        score = 0.0
    word = word.strip('\n')
    style = {'margin-left': '1px', 'margin-right': '2px', 'background-color': getRGB(colormap(score))}
    span = html.Span([word], style = style)
    return span

def dash_html_prediction_score(words, scores,cmap_name="bwr"):
 """
 Return word-level heatmap in HTML format,
 with words being the list of words (as string),
 scores the corresponding list of word-level relevance values,
 and cmap_name the name of the matplotlib diverging colormap.
 Refer to https://matplotlib.org/stable/tutorials/colors/colormaps.html for cmap_name
 'Greens' for prediction, 'bwr' for relevance
 """

 colormap = plt.get_cmap(cmap_name)

 #try:
 # assert len(words) < len(scores)
 #except:
 # print(len(words), len(scores))
 try:
  max_s = max(scores)
 except:
  print("No max score")
  max_s = 0.5
  scores = [0.5] * len(words)

 try:
  min_s = min(scores)
 except:
  print("No min score")
  min_s = 0.5
  scores = [0.5] * len(words)
  

 output_text_div_children = [] 
 scaled_scores = []
 for idx, w in enumerate(words):
      #w = w.replace('<','(').replace('>',')')
      #if max_s > 0.5:
        #score = rescale_score_by_abs(scores[idx], max_s, min_s)
      #elif max_s < 0.1:
      #    score = 0.5+scores[idx]
      
      if scores[idx] >= 0.5:
          # range between 0.6 and 0.9
          score = 0.5 + scores[idx]/2
      elif  0.1 <= scores[idx] < 0.5:
          score = 0.5 + scores[idx]/3
      else:
          score = 0.5 + scores[idx]
      #scaled_scores.append(score)
      #output_text = output_text + table_row(span_word(w, score, colormap))
      w_span = dash_span_word(w, score, colormap)
      output_text_div_children.append(w_span)

 return output_text_div_children

def dash_html_prediction_score_annotate(words, scores,cmap_name="bwr"):
 """
 Return word-level heatmap in HTML format,
 with words being the list of words (as string),
 scores the corresponding list of word-level relevance values,
 and cmap_name the name of the matplotlib diverging colormap.
 Refer to https://matplotlib.org/stable/tutorials/colors/colormaps.html for cmap_name
 'Greens' for prediction, 'bwr' for relevance
 """

 colormap = plt.get_cmap(cmap_name)

 #try:
 # assert len(words) < len(scores)
 #except:
 # print(len(words), len(scores))
 try:
  max_s = max(scores)
 except:
  print("No max score")
  max_s = 0.5
  scores = [0.5] * len(words)

 try:
  min_s = min(scores)
 except:
  print("No min score")
  min_s = 0.5
  scores = [0.5] * len(words)
  

 output_text_div_children = [] 
 scaled_scores = []
 for idx, w in enumerate(words):
      #w = w.replace('<','(').replace('>',')')
      #if max_s > 0.5:
        #score = rescale_score_by_abs(scores[idx], max_s, min_s)
      #elif max_s < 0.1:
      #    score = 0.5+scores[idx]
      
      if scores[idx] >= 0.5:
          # range between 0.6 and 0.9
          score = 0.5 + scores[idx]/2
      elif  0.1 <= scores[idx] < 0.5:
          score = 0.5 + scores[idx]/3
      else:
          score = 0.5 + scores[idx]
      #scaled_scores.append(score)
      #output_text = output_text + table_row(span_word(w, score, colormap))
      w_span = dash_span_word(w, score, colormap)
      output_text_div_children.append(w_span)

 return output_text_div_children
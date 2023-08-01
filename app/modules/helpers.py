import os
import yaml
from torch import nn, Tensor
import torch
import random
import numpy as np
import collections
import matplotlib.pyplot as plt

def robust_decode(bs):
    '''Takes a byte string as param and convert it into a unicode one.
First tries UTF8, and fallback to Latin1 if it fails'''
    cr = None
    try:
        cr = bs.decode('utf-8')
    except UnicodeDecodeError:
        cr = bs.decode('latin-1')
    return cr

def robust_encode(bs):
    '''Encode a string as param and convert it into a unicode one.
First tries UTF8, and fallback to Latin1 if it fails'''
    cr = None
    try:
        cr = bs.encode('utf-8')
    except UnicodeEncodeError:
        cr = bs.encode('latin-1')
    return cr

def remove_umlaut(string):
    string = string.replace('ü', 'ue')
    string = string.replace('Ü', 'Ue')
    string = string.replace('ä' ,'ae')
    string = string.replace('Ä', 'Ae')
    string = string.replace('ö', 'oe')
    string = string.replace('Ö', 'Oe')
    string = string.replace('ß', 'ss')
    return string

def revert_umlaut(string):
    string = string.replace('ue', 'ü')
    string = string.replace('Ue','Ü')
    string = string.replace('ae','ä')
    string = string.replace('Ae', 'Ä')
    string = string.replace('oe', 'ö')
    string = string.replace('Oe', 'Ö')
    #string = string.replace('ss', 'ß')
    return string

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
   

def _nonlin_fct(nonlin):
    if nonlin == "tanh":
        return torch.tanh
    elif nonlin == "relu":
        return torch.relu
    elif nonlin == "gelu":
        return nn.functional.gelu
    elif nonlin == "sigmoid":
        return torch.sigmoid
    else:
        raise ValueError("Unsupported nonlinearity!")

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-9):
        super().__init__()
        self.mean_square_error = nn.MSELoss(reduction = 'mean')
        self.eps = eps
        
    def forward(self,x,y):
        loss = torch.sqrt(self.mean_square_error(x,y) + self.eps)
        return loss

def find_best_checkpoint(checkpoint_files):
    files = [f.split('=') for f in checkpoint_files]
    scores_and_files = {p[-1].strip('.ckpt'): '='.join(p) for p in files}
    scores_sorted=sorted(scores_and_files.keys())
    return scores_and_files[scores_sorted[0]]

def find_best_checkpoint_path(checkpoint_paths):
    list_to_dict = {}
    for path in checkpoint_paths:
        ckpts = os.listdir(path)
        for ckpt in ckpts:
            ckpt_path = os.path.join(path, ckpt)
            ckpt_loss = ckpt.split('=')[-1].strip('.ckpt')
            list_to_dict[ckpt_loss] = ckpt_path
    best_ckpt_key =  sorted(list(list_to_dict.keys()))[0]
    return list_to_dict[best_ckpt_key] 
    

def load_config(path="configs/default.yaml") -> dict:
    """
    Loads and parses a YAML configuration file.

    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    with open(path, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg

def tile(x: Tensor, count: int, dim=0) -> Tensor:
    """
    Tiles x on dimension dim count times. From OpenNMT. Originally used for beam search.

    :param x: tensor to tile
    :param count: number of tiles
    :param dim: dimension along which the tensor is tiled
    :return: tiled tensor
    """
    if isinstance(x, tuple):
        h, c = x
        return tile(h, count, dim=dim), tile(c, count, dim=dim)

    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
        .transpose(0, 1) \
        .repeat(count, 1) \
        .transpose(0, 1) \
        .contiguous() \
        .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x

def set_seed(seed: int) -> None:
    """
    Set the random seed for modules torch, numpy and random.

    :param seed: random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)

def build_combine_vocab(eng_vocab, ger_vocab, eng_tokenizer, ger_tokenizer, check = False):  
    
    # German BERT has bigger vocab than English BERT, add the English vocab to the German tokenizer
    # eng_vocab = eng_tokenizer.vocab
    # ger_vocab = ger_tokenizer.vocab
    
    # If need to check the intersection between two vocab again for new inital BERT models
    added_vocab = set(eng_vocab.keys()) - set(ger_vocab.keys())
    if check:
        eng_tokenizer.add_tokens(list(ger_vocab.keys()))
        ger_tokenizer.add_tokens(list(eng_vocab.keys()))
        
        if len(eng_tokenizer.vocab) > len(ger_tokenizer.vocab):
            return eng_tokenizer, added_vocab, 'English'
        else:
            return ger_tokenizer, added_vocab, 'German'
    
    
    else:
        ger_tokenizer.add_tokens(list(eng_vocab.keys()))
        return ger_tokenizer, added_vocab, 'German'
    
    
def copy_word_embeddings(eng_embeddings, ger_embeddings, added_vocab, origin_vocab, extended_vocab, retain='German'):
    
    # Initial the resized German embeddings weight with weight from English embeddings
    if retain == 'German':
        embeddings_to_use = ger_embeddings
        embeddings_to_drop = eng_embeddings
    else:
        embeddings_to_use = eng_embeddings
        embeddings_to_drop = ger_embeddings
    
    
    assert embeddings_to_use.word_embeddings.weight.size(0) == len(extended_vocab), "The embeddings for further usage {} must be resized to be same as the length {} of the extended vocabulary.".format(embeddings_to_use.word_embeddings.weight.size(0), len(extended_vocab))
        
    #with torch.no_grad():
    for w in added_vocab:
            
            try:
                #print(w, extended_vocab[w], origin_vocab[w])
                new_idx = extended_vocab[w]
                old_idx = origin_vocab[w]
                embeddings_to_use.word_embeddings.weight[new_idx].data = embeddings_to_drop.word_embeddings.weight[old_idx].detach().clone()
            except:
                print("Token {} is not found in the vocab.".format(w))
            
                
# Average the checkpoints of different seed or different sampling 
def average_checkpoints(checkpoint_paths):
    """Loads checkpoints and returns a model with averaged weights.
    Args:
      inputs: An iterable of string paths of checkpoints to load from.
    Returns:
      A dict of string keys mapping to various values. The 'model' key
      from the returned dict should correspond to an OrderedDict mapping
      string parameter names to torch Tensors.
    """
    params_dict = collections.OrderedDict()
    params_keys = None
    new_state = None
    #checkpoints = os.listdir(checkpoint_dir_path)
    #checkpoint_paths = [os.path.join(checkpoint_dir_path, c) for c in checkpoints]
    num_models = len(checkpoint_paths)

    for fpath in checkpoint_paths:
        with open(fpath, "rb") as f:
            state = torch.load(
                f,
                map_location=(
                    lambda s, _: torch.serialization.default_restore_location(s, "cpu")
                ),
            )
        # Copies over the settings from the first checkpoint
        if new_state is None:
            new_state = state
        
        print(state.keys())
        model_params = state["state_dict"]

        model_params_keys = list(model_params.keys())
        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError(
                "For checkpoint {}, expected list of params: {}, "
                "but found: {}".format(f, params_keys, model_params_keys)
            )

        for k in params_keys:
            p = model_params[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if k not in params_dict:
                params_dict[k] = p.clone()
                # NOTE: clone() is needed in case of p is a shared parameter
            else:
                params_dict[k] += p

    averaged_params = collections.OrderedDict()
    for k, v in params_dict.items():
        averaged_params[k] = v
        if averaged_params[k].is_floating_point():
            averaged_params[k].div_(num_models)
        else:
            averaged_params[k] //= num_models
    new_state["state_dict"] = averaged_params
    return new_state


# Heatmap utils
def getRGB(c_tuple):
 return "#%02x%02x%02x" % (int(c_tuple[0] * 255), int(c_tuple[1] * 255), int(c_tuple[2] * 255))


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


def span_word(word, score, colormap):
 if score == None:
  score = 0.5
 return "<span style=\"background-color:" + getRGB(colormap(score)) + "\">" + word + "</span>"

def relevance_heatmap(words, scores,cmap_name="bwr"):
 """
 Return word-level heatmap in HTML format,
 with words being the list of words (as string),
 scores the corresponding list of word-level relevance values,
 and cmap_name the name of the matplotlib diverging colormap.
 """

 colormap = plt.get_cmap(cmap_name)

 #try:
 #  assert len(words) < len(scores)
 #except:
 # print(len(words), len(scores))
 try:
  max_s = max(scores)
 except:
  print("No max score")
  max_s = 0
  scores = [0] * len(words)

 try:
  min_s = min(scores)
 except:
  print("No min score")
  min_s = 0
  scores = [0] * len(words)

 output_text = []

 for  w ,s in zip(words, scores):
    if s > 0.1:
        score = rescale_score_by_abs(s, max_s, min_s)
        output_text.append(span_word(w, score, colormap))
    else:
        output_text.append(w)
 return ' '.join(output_text)


def html_heatmap(words, scores,cmap_name="bwr"):
 """
 Return word-level heatmap in HTML format,
 with words being the list of words (as string),
 scores the corresponding list of word-level relevance values,
 and cmap_name the name of the matplotlib diverging colormap.
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
  max_s = 0
  scores = [0] * len(words)

 try:
  min_s = min(scores)
 except:
  print("No min score")
  min_s = 0
  scores = [0] * len(words)

 output_text = ""
 scaled_scores = []
 for idx, w in enumerate(words):
  score = rescale_score_by_abs(scores[idx], max_s, min_s)
  scaled_scores.append(score)
  output_text = output_text + table_row(span_word(w, score, colormap))

 return output_text, scaled_scores


def table_row(span_word):
 return '<td class="tg-0pky">{}</td>\n'.format(span_word)


def html_table(words, scores_matrix):
 # list of words, list of scores
 assert len(scores_matrix)  == len(words)
 # colormap  = plt.get_cmap(cmap_name)
 # assert len(words)== len(scores_matrix)
 output_text = '<table class="tg">\n '
 for i in range(len(words)):
  #output_text += '<tr><th class="tg-0pky">Retrieval score: {}</th>\n'.format( str(relevance[i])[:5])

  output_text += html_heatmap(words[i], scores_matrix[i])
  output_text += '</tr>\n'

 output_text += "</table>\n"
 return output_text

"""
Override threading custom module
"""
import sys
import threading
import socket

def get_host_name():
    """
    Get the url of the current host
    Returns
    -------
    String
        host name
    """
    return socket.gethostname()


class CustomThread(threading.Thread):
    """
    Python ovveride threading class
    Used to kill a thread from python object
    Parameters
    ----------
    threading : threading.Thread
        Thread which you want to instanciate
    """
    def __init__(self, *args, **keywords):
        threading.Thread.__init__(self, *args, **keywords)
        self.killed = False
        self.__run_backup = None

    def start(self):
        self.__run_backup = self.run
        self.run = self.__run
        threading.Thread.start(self)


    def __run(self):
        sys.settrace(self.globaltrace)
        self.__run_backup()
        self.run = self.__run_backup

    def globaltrace(self, frame, event, arg):
        """
        Track the global trace
        """
        if event == 'call':
            return self.localtrace
        else:
            return None

    def localtrace(self, frame, event, arg):
        """
        Track the local trace
        """
        if self.killed:
            if event == 'line':
                raise SystemExit()
        return self.localtrace

    def kill(self):
        """
        Kill the current Thread
        """
        self.killed = True
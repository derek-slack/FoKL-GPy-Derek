from preprocessing import *
from sampler import * 
from postprocessing import *
from FoKL_Function import *


def load_model(filename, directory=None): 
    return load(filename, directory)

def _str_to_bool(s): 
    return str_to_bool(s)

def _process_kwargs(**kwargs):
    return process_kwargs(**kwargs)

def _set_attributes(self, **kwargs):
    return set_attributes(self, **kwargs)

def _merge_dicts(dict1, dict2):
    return merge_dicts(dict1, dict2)

class FoKL: 
    def __init__(self, **kwargs):
        
from .utils import str_to_bool, process_kwargs, merge_dicts, set_attributes
from .config import FoKLConfig
from .fokl_to_pyomo import fokl_to_pyomo
from .preprocessing.kernels import getKernels
from .preprocessing.dataFormat import dataFormat
from .sampler.samplers import fitSampler
from .postprocessing.postprocessing import postprocess
from .FoKL_Function.Functions import Functions
import warnings
import numpy as np


class FoKL:
    def __init__(self, **kwargs):
        self.config = FoKLConfig()
        self.dataFormat = dataFormat(self, self.config)
        self.functions = Functions(self, self.config, self.dataFormat)
        self.fitSampler = fitSampler(self, self.config, self.dataFormat, self.functions)
        self.postprocessing = postprocess(self, self.config, self.dataFormat, self.functions)

        current = process_kwargs(self.config.DEFAULT, kwargs) # = default, but updated by any user kwargs
        for boolean in ['gimmie', 'way3', 'aic', 'UserWarnings', 'ConsoleOutput']:
            if not (current[boolean] is False or current[boolean] is True): 
                current[boolean] = str_to_bool(current[boolean])

        # Load spline coefficients:
        phis = current['phis']  # in case advanced user is testing other splines
        if isinstance(current['kernel'], int):  # then assume integer indexing 'self.kernels'
            current['kernel'] = self.config.KERNELS[current['kernel']]  # update integer to string
        if current['phis'] is None: # if default
            if current['kernel'] == self.config.KERNELS[0]: # == 'Cubic Splines':
                current['phis'] = getKernels.sp500()
            elif current['kernel'] == self.config.KERNELS[1]:   # == 'Bernoulli Polynomials':
                current['phis'] = getKernels.bernoulli()
            elif isinstance(current['kernel'], str):    # confirm string before printing to console
                raise ValueError(f"The user-provided kernel '{current['phis']}' is not supported.")
            else:
                raise ValueError(f"The user-provided kernel is not supported.")
            
        if current['UserWarnings']:
            warnings.filterwarnings("default", category=UserWarning)
        else: 
            warnings.filterwarnings("ignore", category=UserWarning)

        for key, value, in current.items():
            setattr(self, key, value)
            
        
            
    # def format(self, inputs, data=None, auto_transpose=True, single_instance=False, bit=64):
    #     return self.dataFormat.format(self, inputs, data, auto_transpose, single_instance, bit)
    
    # def normalize(self, inputs, minmax=None, pillow=None, pillow_type='percent'):
    #     return self.dataFormat.normalize(self, inputs, minmax, pillow, pillow_type)
    
    # def clean(self, inputs, data=None, AutoTranspose=True, SingleInstance=False, bit=64):
    #     inputs, data = self.dataFormat.clean(self, inputs, data, AutoTranspose, SingleInstance, bit)
    #     if data is None: 
    #         return inputs
    #     else: 
    #         return inputs, data
    
    # def generate_trainlog(self, train, n=None):
    #     return self.dataFormat.generate_trainlog(self, train, n)
    
    def coverage3(self, **kwargs): 
        return self.postprocessing.coverage3(**kwargs)
    
    def fit(self, inputs=None, data=None, **kwargs):
        dataFormat.inputs = inputs 
        self.inputs, self.data, self.betas, self.minmax, self.mtx, evs = self.fitSampler.fit(inputs, data, **kwargs)
        
        # postprocess.inputs = inputs
        # postprocess.data = data
        # postprocess.betas = betas
        # postprocess.mtx = mtx
        return self.betas, self.mtx, self.minmax, evs
    
    # need to do more examination 
    def clear(self, keep=None, clear=None, all=False):
        return clear(keep=keep, clear=clear, all=all)
    
    def to_pyomo(self, xvars, yvars, m=None, xfix=None, yfix=None, truescale=True, std=True, draws=None):
        return fokl_to_pyomo(self, xvars, yvars, m, xfix, yfix, truescale, std, draws)
    
    def save(self, filename=None, **kwargs):
        return self.functions.save(filename, **kwargs)
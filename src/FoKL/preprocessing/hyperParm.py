__all__ = ['_format', '_merge_dicts', '_set_attributes', '_str_to_bool']
from FoKL import getKernels
from kwargProcessing import _str_to_bool, _process_kwargs
import numpy as np 
import panda as pd 
import  warnings 

def __init__(self, **kwargs):
    """
    Initialization Inputs (i.e., hyperparameters and their descriptions):
        - 'kernel' is a string defining the kernel to use for building the model, which defines 'phis', a data
        structure with coefficients for the basis functions.
            - If set to 'Cubic Splines', then 'phis' defines 500 splines (i.e., basis functions) of 499 piecewise
            cubic polynomials each. (from 'splineCoefficient500_highPrecision_smoothed.txt').
                - y = sum(phis[spline_index][k][piecewise_index] * (x ** k) for k in range(4))
            - If set to 'Bernoulli Polynomials', then 'phis' defines the first 258 non-zero Bernoulli polynomials 
            (i.e., basis functions). (from 'bernoulliNumbers258.txt').
                - y = sum(phis[polynomial_index][k] * (x ** k) for k in range(len(phis[polynomial_index])))
        - 'phis' gets defined automatically by 'kernel', but if testing other coefficients with the same format
        implied by 'kernel' then 'phis' may be user-defined.
        - 'relats_in' is a boolean matrix indicating which terms should be excluded from the model building. For
        instance, if a certain main effect should be excluded 'relats_in' will include a row with a 1 in the column
        for that input and zeros elsewhere. If a certain two-way interaction should be excluded there should be a
        row with ones in those columns and zeros elsewhere. To exclude no terms, leave blank. For an example of
        excluding the first input main effect and its interaction with the third input for a case with three total
        inputs, 'relats_in = [[1, 0, 0], [1, 0, 1]]'.
        - 'a' and 'b' are the shape and scale parameters of the ig distribution for the observation error variance
        of the data. The observation error model is white noise. Choose the mode of the ig distribution to match the
        noise in the output dataset and the mean to broaden it some.
        - 'atau' and 'btau' are the parameters of the ig distribution for the 'tau squared' parameter: the variance
        of the beta priors is iid normal mean zero with variance equal to sigma squared times tau squared. Tau
        squared must be scaled in the prior such that the product of tau squared and sigma squared scales with the
        output dataset.
        - 'tolerance' controls how hard the function builder tries to find a better model once adding terms starts
        to show diminishing returns. A good default is 3, but large datasets could benefit from higher values.
        - 'burnin' is the total number of draws from the posterior for each tested model before the 'draws' draws.
        - 'draws' is the total number of draws from the posterior for each tested model after the 'burnin' draws.
        There draws are what appear in 'betas' after calling 'fit', and the 'burnin' draws are discarded.
        - 'gimmie' is a boolean causing the routine to return the most complex model tried instead of the model with
        the optimum bic.
        - 'way3' is a boolean specifying the calculation of three-way interactions.
        - 'threshav' and 'threshstda' form a threshold for the elimination of terms.
            - 'threshav' is a threshold for proposing terms for elimination based on their mean values, where larger
            thresholds lead to more elimination.
            - 'threshstda' is a threshold standard deviation expressed as a fraction relative to the mean.
            - terms with coefficients that are lower than 'threshav' and higher than 'threshstda' will be proposed
            for elimination but only executed based on relative BIC values.
        - 'threshstdb' is a threshold standard deviation that is independent of the mean value of the coefficient.
        All terms with a standard deviation (relative to mean) exceeding this will be proposed for elimination.
        - 'aic' is a boolean specifying the use of the aikaike information criterion.
    Default Values for Hyperparameters:
        - kernel     = 'Cubic Splines'
        - phis       = f(kernel)
        - relats_in  = []
        - a          = 4
        - b          = f(a, data)
        - atau       = 4
        - btau       = f(atau, data)
        - tolerance  = 3
        - burnin     = 1000
        - draws      = 1000
        - gimmie     = False
        - way3       = False
        - threshav   = 0.05
        - threshstda = 0.5
        - threshstdb = 2
        - aic        = False
    Other Optional Inputs:
        - UserWarnings  == boolean to print user-warnings to the command terminal          == True (default)
        - ConsoleOutput == boolean to print [ind, ev] during 'fit' to the command terminal == True (default)
    """
    # Store list of hyperparameters for easy reference later, if sweeping through values in functions such as fit:
    self.hypers = ['kernel', 'phis', 'relats_in', 'a', 'b', 'atau', 'btau', 'tolerance', 'burnin', 'draws',
                   'gimmie', 'way3', 'threshav', 'threshstda', 'threshstdb', 'aic', 'update', 'built']
    # Store list of settings for easy reference later (namely, in 'clear'):
    self.settings = ['UserWarnings', 'ConsoleOutput']
    # Store supported kernels for later logical checks against 'kernel':
    self.kernels = ['Cubic Splines', 'Bernoulli Polynomials']
    # List of attributes to keep in event of clearing model (i.e., 'self.clear'):
    self.keep = ['keep', 'hypers', 'settings', 'kernels'] + self.hypers + self.settings + self.kernels
    # Process user's keyword arguments:
    default = {
               # Hyperparameters:
               'kernel': 'Cubic Splines', 'phis': None, 'relats_in': [], 'a': 4, 'b': None, 'atau': 4,
               'btau': None, 'tolerance': 3, 'burnin': 1000, 'draws': 1000, 'gimmie': False, 'way3': False,
               'threshav': 0.05, 'threshstda': 0.5, 'threshstdb': 2, 'aic': False,
                # Hyperparameters with Update:
                'sigsqd0': 0.5, 'burn': 500, 'update': False, 'built' : False,
               # Other:
               'UserWarnings': True, 'ConsoleOutput': True
               }
    current = _process_kwargs(default, kwargs)  # = default, but updated by any user kwargs
    for boolean in ['gimmie', 'way3', 'aic', 'UserWarnings', 'ConsoleOutput']:
        if not (current[boolean] is False or current[boolean] is True):
            current[boolean] = _str_to_bool(current[boolean])
    # Load spline coefficients:
    phis = current['phis']  # in case advanced user is testing other splines
    if isinstance(current['kernel'], int):  # then assume integer indexing 'self.kernels'
        current['kernel'] = self.kernels[current['kernel']]  # update integer to string
    if current['phis'] is None:  # if default
        if current['kernel'] == self.kernels[0]:  # == 'Cubic Splines':
            current['phis'] = getKernels.sp500()
        elif current['kernel'] == self.kernels[1]:  # == 'Bernoulli Polynomials':
            current['phis'] = getKernels.bernoulli()
        elif isinstance(current['kernel'], str):  # confirm string before printing to console
            raise ValueError(f"The user-provided kernel '{current['phis']}' is not supported.")
        else:
            raise ValueError(f"The user-provided kernel is not supported.")
    # Turn on/off FoKL warnings:
    if current['UserWarnings']:
        warnings.filterwarnings("default", category=UserWarning)
    else:
        warnings.filterwarnings("ignore", category=UserWarning)
    # Store values as class attributes:
    for key, value in current.items():
        setattr(self, key, value)

            
def _format(self, inputs, data=None, AutoTranspose=True, SingleInstance=False, bit=64):
   """
   Called by 'clean' to format dataset.
       - formats inputs as 2D ndarray, where columns are input variables; n_rows > n_cols if AutoTranspose=True
       - formats data as 2D ndarray, with single column   Note SingleInstance has priority over AutoTranspose. If SingleInstance=True, then AutoTranspose=False.
   """
   # Format and check inputs:
   AutoTranspose = _str_to_bool(AutoTranspose)
   SingleInstance = _str_to_bool(SingleInstance)
   bits = {16: np.float16, 32: np.float32, 64: np.float64}  # allowable datatypes: https://numpy.org/doc/stable/reference/arrays.scalars.html#arrays-scalars-built-in
   if SingleInstance is True:
       AutoTranspose = False
   if bit not in bits.keys():
       warnings.warn(f"Keyword 'bit={bit}' limited to values of 16, 32, or 64. Assuming default value of 64.", category=UserWarning)
       bit = 64
   datatype = bits[bit]   # Convert 'inputs' and 'data' to numpy if pandas:
   if any(isinstance(inputs, type) for type in (pd.DataFrame, pd.Series)):
       inputs = inputs.to_numpy()
       warnings.warn("'inputs' was auto-converted to numpy. Convert manually for assured accuracy.",
                     category=UserWarning)
   if data is not None:
       if any(isinstance(data, type) for type in (pd.DataFrame, pd.Series)):
           data = data.to_numpy()
           warnings.warn("'data' was auto-converted to numpy. Convert manually for assured accuracy.",
                         category=UserWarning)   # Format 'inputs' as [n x m] numpy array:
   inputs = np.array(inputs)  # attempts to handle lists or any other format (i.e., not pandas)
   if inputs.ndim > 2:  # remove axes with 1D for cases like (N x 1 x M) --> (N x M)
       inputs = np.squeeze(inputs)
   if inputs.dtype != datatype:
       inputs = np.array(inputs, dtype=datatype)
       warnings.warn(f"'inputs' was converted to float{bit}. May require user-confirmation that "
                     f"values did not get corrupted.", category=UserWarning)
   if inputs.ndim == 1:  # if inputs.shape == (number,) != (number,1), then add new axis to match FoKL format
       if SingleInstance is True:
           inputs = inputs[np.newaxis, :]  # make 1D into (1, M)
       else:
           inputs = inputs[:, np.newaxis]  # make 1D into (N, 1)
   if AutoTranspose is True and SingleInstance is False:
       if inputs.shape[1] > inputs.shape[0]:  # assume user is using transpose of proper format
           inputs = inputs.transpose()
           warnings.warn("'inputs' was transposed. Ignore if more datapoints than input variables, else set "
                         "'AutoTranspose=False' to disable.", category=UserWarning)   # Format 'data' as [n x 1] numpy array:
   if data is not None:
       data = np.array(data)  # attempts to handle lists or any other format (i.e., not pandas)
       data = np.squeeze(data)
       if data.dtype != datatype:
           data = np.array(data, dtype=datatype)
           warnings.warn(f"'data' was converted to float{bit}. May require user-confirmation that "
                         f"values did not get corrupted.", category=UserWarning)
       if data.ndim == 1:  # if data.shape == (number,) != (number,1), then add new axis to match FoKL format
           data = data[:, np.newaxis]
       else:  # check user provided only one output column/row, then transpose if needed
           n = data.shape[0]
           m = data.shape[1]
           if (m != 1 and n != 1) or (m == 1 and n == 1):
               raise ValueError("Error: 'data' must be a vector.")
           elif m != 1 and n == 1:
               data = data.transpose()
               warnings.warn("'data' was transposed to match FoKL formatting.", category=UserWarning)
           
   return inputs, data


from ..utils import str_to_bool, process_kwargs
import numpy as np  
import panda as pd 
import  warnings 
            
def format(self, inputs, data=None, AutoTranspose=True, SingleInstance=False, bit=64):
   """
   Called by 'clean' to format dataset.
       - formats inputs as 2D ndarray, where columns are input variables; n_rows > n_cols if AutoTranspose=True
       - formats data as 2D ndarray, with single column   Note SingleInstance has priority over AutoTranspose. If SingleInstance=True, then AutoTranspose=False.
   """
   # Format and check inputs:
   AutoTranspose = str_to_bool(AutoTranspose)
   SingleInstance = str_to_bool(SingleInstance)
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


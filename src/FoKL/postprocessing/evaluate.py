from FoKLRoutines_update import FoKL
import numpy as np 
import warnings
from ..utils import str_to_bool, process_kwargs, merge_dicts, set_attributes

class evaluate(FoKL):
    def evaluate(self, inputs=None, betas=None, mtx=None, **kwargs):
        """
        Evaluate the FoKL model for provided inputs and (optionally) calculate bounds. Note 'evaluate_fokl' may be a
        more accurate name so as not to confuse this function with 'evaluate_basis', but it is unexpected for a user to
        call 'evaluate_basis' so this function is simply named 'evaluate'.
        Input:
            inputs == input variable(s) at which to evaluate the FoKL model == self.inputs (default)
        Optional Inputs:
            betas        == coefficients defining FoKL model                       == self.betas (default)
            mtx          == interaction matrix defining FoKL model                 == self.mtx (default)
            minmax       == [min, max] of inputs used for normalization            == None (default)
            draws        == number of beta terms used                              == self.draws (default)
            clean        == boolean to automatically normalize and format 'inputs' == False (default)
            ReturnBounds == boolean to return confidence bounds as second output   == False (default)
        """
        # Process keywords:
        default = {'minmax': None, 'draws': self.draws, 'clean': False, 'ReturnBounds': False,  # for evaluate
                   '_suppress_normalization_warning': False}                                    # if called from coverage3
        default_for_clean = {'train': 1, 
                             # For '_format':
                             'AutoTranspose': True, 'SingleInstance': False, 'bit': 64,
                             # For '_normalize':
                             'normalize': True, 'minmax': None, 'pillow': None, 'pillow_type': 'percent'}
        current = process_kwargs(merge_dicts(default, default_for_clean), kwargs)
        for boolean in ['clean', 'ReturnBounds']:
            current[boolean] = str_to_bool(current[boolean])
        kwargs_to_clean = {}
        for kwarg in default_for_clean.keys():
            kwargs_to_clean.update({kwarg: current[kwarg]})  # store kwarg for clean here
            del current[kwarg]  # delete kwarg for clean from current
        if current['draws'] < 40 and current['ReturnBounds']:
            current['draws'] = 40
            warnings.warn("'draws' must be greater than or equal to 40 if calculating bounds. Setting 'draws=40'.")
        draws = current['draws']  # define local variable
        if betas is None:  # default
            if draws > self.betas.shape[0]:
                draws = self.betas.shape[0]  # more draws than models results in inf time, so threshold
                self.draws = draws
                warnings.warn("Updated attribute 'self.draws' to equal number of draws in 'self.betas'.",
                              category=UserWarning)
            betas = self.betas[-draws::, :]  # use samples from last models
        else:  # user-defined betas may need to be formatted
            betas = np.array(betas)
            if betas.ndim == 1:
                betas = betas[np.newaxis, :]  # note transpose would be column of beta0 terms, so not expected
            if draws > betas.shape[0]:
                draws = betas.shape[0]  # more draws than models results in inf time, so threshold
            betas = betas[-draws::, :]  # use samples from last models
        if mtx is None:  # default
            mtx = self.mtx
        else:  # user-defined mtx may need to be formatted
            if isinstance(mtx, int):
                mtx = [mtx]
            mtx = np.array(mtx)
            if mtx.ndim == 1:
                mtx = mtx[np.newaxis, :]
                warnings.warn("Assuming 'mtx' represents a single model. If meant to represent several models, then "
                              "explicitly enter a 2D numpy array where rows correspond to models.")
        phis = self.phis
        # Automatically normalize and format inputs:
        if inputs is None:  # default
            inputs = self.inputs
            if current['clean']:
                warnings.warn("Cleaning was already performed on default 'inputs', so overriding 'clean' to False.",
                              category=UserWarning)
                current['clean'] = False
        else:  # user-defined 'inputs'
            if not current['clean']:  # assume provided inputs are already formatted and normalized
                normputs = inputs
                if current['_suppress_normalization_warning'] is False:  # to suppress warning when evaluate called from coverage3
                    warnings.warn("User-provided 'inputs' but 'clean=False'. Subsequent errors may be solved by enabling automatic formatting and normalization of 'inputs' via 'clean=True'.", category=UserWarning)
        if current['clean']:
            normputs = self.clean(inputs, kwargs_from_other=kwargs_to_clean)
        elif inputs is None:
            normputs = self.inputs
        else:
            normputs = np.array(inputs)
        m, mbets = np.shape(betas)  # Size of betas
        n, mputs = np.shape(normputs)  # Size of normalized inputs
        setnos_p = np.random.randint(m, size=(1, draws))  # Random draws  from integer distribution
        i = 1
        while i == 1:
            setnos = np.unique(setnos_p)
            if np.size(setnos) == np.size(setnos_p):
                i = 0
            else:
                setnos_p = np.append(setnos, np.random.randint(m, size=(1, draws - np.shape(setnos)[0])))
        X = np.zeros((n, mbets))
        normputs = np.asarray(normputs)
        if self.kernel == self.kernels[0]:  # == 'Cubic Splines':
            _, phind, xsm = self._inputs_to_phind(normputs)  # ..., phis=self.phis, kernel=self.kernel) already true
        elif self.kernel == self.kernels[1]:  # == 'Bernoulli Polynomials':
            phind = None
            xsm = normputs
        for i in range(n):
            for j in range(1, mbets):
                phi = 1
                for k in range(mputs):
                    num = mtx[j - 1, k]
                    if num > 0:
                        nid = int(num - 1)
                        if self.kernel == self.kernels[0]:  # == 'Cubic Splines':
                            coeffs = [phis[nid][order][phind[i, k]] for order in range(4)]  # coefficients for cubic
                        elif self.kernel == self.kernels[1]:  # == 'Bernoulli Polynomials':
                            coeffs = phis[nid]  # coefficients for bernoulli
                        phi *= self.evaluate_basis(coeffs, xsm[i, k])  # multiplies phi(x0)*phi(x1)*etc.
                X[i, j] = phi
        X[:, 0] = np.ones((n,))
        modells = np.zeros((n, draws))  # note n == np.shape(data)[0] if data != 'ignore'
        for i in range(draws):
            modells[:, i] = np.transpose(np.matmul(X, np.transpose(np.array(betas[setnos[i], :]))))
        mean = np.mean(modells, 1)
        if current['ReturnBounds']:
            bounds = np.zeros((n, 2))  # note n == np.shape(data)[0] if data != 'ignore'
            cut = int(np.floor(draws * 0.025) + 1)
            for i in range(n):  # note n == np.shape(data)[0] if data != 'ignore'
                drawset = np.sort(modells[i, :])
                bounds[i, 0] = drawset[cut]
                bounds[i, 1] = drawset[draws - cut]
            return mean, bounds
        else:
            return mean
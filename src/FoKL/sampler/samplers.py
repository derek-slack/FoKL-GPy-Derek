__all__ = ["BaseSampler", "GibbsSampler", "GibbsUpdateSampler"]
import gibbs, gibbs_Xin_update
from abc import ABC, abstractmethod
from preprocessing.dataFormat import clean
from FoKL_Function import trainset
from ..utils import process_kwargs, str_to_bool
import warnings
import numpy as np
import math
import itertools

class BaseSampler(ABC):
    def __init__(self, inputs, data, phis, Xin, discmtx, a, b, atau, btau, draws, phind, xsm, sigsqd, tausqd, dtd):
        self.inputs = inputs
        self.data = data
        self.phis = phis
        self.Xin = Xin
        self.discmtx = discmtx
        self.a = a
        self.b = b
        self.atau = atau
        self.btau = btau
        self.draws = draws
        self.phind = phind
        self.xsm = xsm
        self.sigsqd = sigsqd
        self.tausqd = tausqd
        self.dtd = dtd

    def fit(self, inputs=None, data=None, **kwargs):
        """
        For fitting model to known inputs and data (i.e., training of model).
        Inputs:
            inputs == NxM matrix of independent (or non-linearly dependent) 'x' variables for fitting f(x1, ..., xM)
            data   == Nx1 vector of dependent variable to create model for predicting the value of f(x1, ..., xM)
        Keyword Inputs (for fit):
            clean         == boolean to perform automatic cleaning and formatting               == False (default)
            ConsoleOutput == boolean to print [ind, ev] to console during FoKL model generation == True (default)
        See 'clean' for additional keyword inputs, which may be entered here.
        Return Outputs:
            'betas' are a draw (after burn-in) from the posterior distribution of coefficients: matrix, with rows
            corresponding to draws and columns corresponding to terms in the GP.
            'mtx' is the basis function interaction matrix from the
            best model: matrix, with rows corresponding to terms in the GP (and thus to the
            columns of 'betas' and columns corresponding to inputs. a given entry in the
            matrix gives the order of the basis function appearing in a given term in the GP.
            all basis functions indicated on a given row are multiplied together.
            a zero indicates no basis function from a given input is present in a given term
            'ev' is a vector of BIC values from all of the models
            evaluated
        Added Attributes:
            - Various ... please see description of 'clean()'
        """


        # Check for unexpected keyword arguments:
        default_for_fit = {'ConsoleOutput': True}
        default_for_fit['ConsoleOutput'] = str_to_bool(kwargs.get('ConsoleOutput', self.ConsoleOutput))
        default_for_fit['clean'] = str_to_bool(kwargs.get('clean', False))
        default_for_clean = {'train': 1, 
                             # For '_format':
                             'AutoTranspose': True, 'SingleInstance': False, 'bit': 64,
                             # For '_normalize':
                             'normalize': True, 'minmax': None, 'pillow': None, 'pillow_type': 'percent'}
        expected = self.hypers + list(default_for_fit.keys()) + list(default_for_clean.keys())
        kwargs = process_kwargs(expected, kwargs)
        if default_for_fit['clean'] is False:
            if any(kwarg in default_for_clean.keys() for kwarg in kwargs.keys()):
                warnings.warn("Keywords for automatic cleaning were defined but clean=False.")
            default_for_clean = {}  # not needed for future since 'clean=False'
        # Process keyword arguments and update/define class attributes:
        kwargs_to_clean = {}
        for kwarg in kwargs.keys():
            if kwarg in self.hypers:  # for case of user sweeping through hyperparameters within 'fit' argument
                if kwarg in ['gimmie', 'way3', 'aic']:
                    setattr(self, kwarg, str_to_bool(kwargs[kwarg]))
                else:
                    setattr(self, kwarg, kwargs[kwarg])
            elif kwarg in default_for_clean.keys():
                # if kwarg in ['']:
                #     kwargs_to_clean.update({kwarg: _str_to_bool(kwargs[kwarg])})
                # else:
                kwargs_to_clean.update({kwarg: kwargs[kwarg]})
        self.ConsoleOutput = default_for_fit['ConsoleOutput']
        # Perform automatic cleaning of 'inputs' and 'data' (unless user specified not to), and handle exceptions:
        error_clean_failed = False
        if default_for_fit['clean'] is True:
            try:
                if inputs is None:  # assume clean already called and len(data) same as train data if data not None
                    inputs, _ = self.trainset()
                if data is None:  # assume clean already called and len(inputs) same as train inputs if inputs not None
                    _, data = self.trainset()
            except Exception as exception:
                error_clean_failed = True
            self.clean(inputs, data, kwargs_from_other=kwargs_to_clean, _setattr=True)
        else:  # user input implies that they already called clean prior to calling fit
            try:
                if inputs is None:  # assume clean already called and len(data) same as train data if data not None
                    inputs, _ = self.trainset()
                if data is None:  # assume clean already called and len(inputs) same as train inputs if inputs not None
                    _, data = self.trainset()
            except Exception as exception:
                warnings.warn("Keyword 'clean' was set to False but is required prior to or during 'fit'. Assuming "
                              "'clean' is True.", category=UserWarning)
                if inputs is None or data is None:
                    error_clean_failed = True
                else:
                    default_for_fit['clean'] = True
                    self.clean(inputs, data, kwargs_from_other=kwargs_to_clean, _setattr=True)
        if error_clean_failed is True:
            raise ValueError("'inputs' and/or 'data' were not provided so 'clean' could not be performed.")
        # After cleaning and/or handling exceptions, define cleaned 'inputs' and 'data' as local variables:
        try:
            inputs, data = self.trainset()
        except Exception as exception:
            warnings.warn("If not calling 'clean' prior to 'fit' or within the argument of 'fit', then this is the "
                          "likely source of any subsequent errors. To troubleshoot, simply include 'clean=True' within "
                          "the argument of 'fit'.", category=UserWarning)
        # Define attributes as local variables:
        phis = self.phis
        relats_in = self.relats_in
        a = self.a
        b = self.b
        atau = self.atau
        btau = self.btau
        tolerance = self.tolerance
        draws = self.burnin + self.draws  # after fitting, the 'burnin' draws will be discarded from 'betas'
        gimmie = self.gimmie
        way3 = self.way3
        threshav = self.threshav
        threshstda = self.threshstda
        threshstdb = self.threshstdb
        aic = self.aic
        # Update 'b' and/or 'btau' if set to default:
        if btau is None or b is None:  # then use 'data' to define (in combination with 'a' and/or 'atau')
            # Calculate variance and mean, both as 64-bit, but for large datasets (i.e., less than 64-bit) be careful
            # to avoid converting the entire 'data' to 64-bit:
            if data.dtype != np.float64:  # and sigmasq == math.inf  # then recalculate but as 64-bit
                sigmasq = 0
                n = data.shape[0]
                data_mean = 0
                for i in range(n):  # element-wise to avoid memory errors when entire 'data' is 64-bit
                    data_mean += np.array(data[i], dtype=np.float64)
                data_mean = data_mean / n
                for i in range(n):  # element-wise to avoid memory errors when entire 'data' is 64-bit
                    sigmasq += (np.array(data[i], dtype=np.float64) - data_mean) ** 2
                sigmasq = sigmasq / (n - 1)
            else:  # same as above but simplified syntax avoiding for loops since 'data.dtype=np.float64'
                sigmasq = np.var(data)
                data_mean = np.mean(data)
            if sigmasq == math.inf:
                warnings.warn("The dataset is too large such that 'sigmasq=inf' even as 64-bit. Consider training on a "
                              "smaller percentage of the dataset.", category=UserWarning)
            if b is None:
                b = sigmasq * (a + 1)
                self.b = b
            if btau is None:
                scale = np.abs(data_mean)
                btau = (scale / sigmasq) * (atau + 1)
                self.btau = btau
        def perms(x):
            """Python equivalent of MATLAB perms."""
            # from https://stackoverflow.com/questions/38130008/python-equivalent-for-matlabs-perms
            a = np.vstack(list(itertools.permutations(x)))[::-1]
            return a
        # Prepare phind and xsm if using cubic splines, else match variable names required for gibbs argument
        if self.kernel == self.kernels[0]:  # == 'Cubic Splines':
            _, phind, xsm = self._inputs_to_phind(inputs)  # ..., phis=self.phis, kernel=self.kernel) already true
        elif self.kernel == self.kernels[1]:  # == 'Bernoulli Polynomials':
            phind = None
            xsm = inputs
        # [BEGIN] initialization of constants (for use in gibbs to avoid repeat large calculations):
        if self.update == True:
            self.betas, self.mtx, self.evs = self.fit(inputs, data)
            return self.betas, self.mtx, self.evs
        # initialize tausqd at the mode of its prior: the inverse of the mode of sigma squared, such that the
        # initial variance for the betas is 1
        sigsqd0 = b / (1 + a)
        tausqd0 = btau / (1 + atau)
        dtd = np.transpose(data).dot(data)
        # Check for large datasets, where 'dtd=inf' is common and causes bug 'betas=nan', by only converting one
        # point to 64-bit at a time since there is likely not enough memory to convert all of 'data' to 64-bit:
        if dtd[0][0] == math.inf and data.dtype != np.float64:
            # # If converting all of 'data' to 64-bit:
            # data64 = np.array(data, dtype=np.float64)  # assuming large dataset means using less than 64-bit
            # dtd = np.dot(data64.T, data64)  # same equation, but now 64-bit
            # Converting one point at a time to limit memory use:
            dtd = 0
            for i in range(data.shape[0]):
                data_i = np.array(data[i], dtype=np.float64)
                dtd += data_i ** 2  # manually calculated inner dot product
            dtd = np.array([dtd])  # to align with dimensions of 'np.transpose(data).dot(data)' such that dtd[0][0]
        if dtd[0][0] == math.inf:
            warnings.warn("The dataset is too large such that the inner product of the output 'data' vector is "
                          "Inf. This will likely cause values in 'betas' to be Nan.", category=UserWarning)
        # [END] initialization of constants

        # 'n' is the number of datapoints whereas 'm' is the number of inputs
        n, m = np.shape(inputs)
        mrel = n
        damtx = np.array([])
        evs = np.array([])
        # Conversion of Lines 79-100 of emulator_Xin.m
        if np.logical_not(all([isinstance(index, int) for index in relats_in])):  # checks if relats is an array
            if np.any(relats_in):
                relats = np.zeros((sum(np.logical_not(relats_in)), m))
                ind = 1
                for i in range(0, m):
                    if np.logical_not(relats_in[i]):
                        relats[ind][i] = 1
                        ind = ind + 1
                ind_in = m + 1
                for i in range(0, m - 1):
                    for j in range(i + 1, m):
                        if np.logical_not(relats_in[ind_in]):
                            relats[ind][i] = 1
                            relats[ind][j] = 1
                            ind = ind + 1
                    ind_in = ind_in + 1
            mrel = sum(np.logical_not(relats_in)).all()
        else:
            mrel = sum(np.logical_not(relats_in))
        # End conversion
        # 'ind' is an integer which controls the development of new terms
        ind = 1
        greater = 0
        finished = 0
        X = []
        killset = []
        killtest = []
        if m == 1:
            sett = 1
        elif way3:
            sett = 3
        else:
            sett = 2
        while True:
            # first we have to come up with all combinations of 'm' integers that
            # sums up to ind
            indvec = np.zeros((m))
            summ = ind
            while summ:
                for j in range(0,sett):
                    indvec[j] = indvec[j] + 1
                    summ = summ - 1
                    if summ == 0:
                        break
            while True:
                vecs = np.unique(perms(indvec),axis=0)
                if ind > 1:
                    mvec, nvec = np.shape(vecs)
                else:
                    mvec = np.shape(vecs)[0]
                    nvec = 1
                killvecs = []
                if mrel != 0:
                    for j in range(1, mvec):
                        testvec = np.divide(vecs[j, :], vecs[j, :])
                        testvec[np.isnan(testvec)] = 0
                        for k in range(1, mrel):
                            if sum(testvec == relats[k, :]) == m:
                                killvecs.append(j)
                                break
                    nuvecs = np.zeros(mvec - np.size(killvecs), m)
                    vecind = 1
                    for j in range(1, mvec):
                        if not (j == killvecs):
                            nuvecs[vecind, :] = vecs[j, :]
                            vecind = vecind + 1
                    vecs = nuvecs
                if ind > 1:
                    vm, vn = np.shape(vecs)
                else:
                    vm = np.shape(vecs)[0]
                    vn = 1
                if np.size(damtx) == 0:
                    damtx = vecs
                else:
                    damtx = np.append(damtx, vecs, axis=0)

                sampler_fn = SAMPLERS_DICT["gibbs"]
                [dam, null] = np.shape(damtx)
                [beters, null, null, null, xers, ev] = sampler_fn(inputs, data, phis, X, damtx, a, b, atau, btau, draws,
                                                             phind, xsm, sigsqd0, tausqd0, dtd)
                if aic:
                    ev = ev + (2 - np.log(n)) * (dam + 1)
                betavs = np.abs(np.mean(beters[int(np.ceil((draws / 2)+1)):draws, (dam - vm + 1):dam+1], axis=0))
                betavs2 = np.divide(np.std(np.array(beters[int(np.ceil(draws/2)+1):draws, dam-vm+1:dam+1]), axis=0),
                    np.abs(np.mean(beters[int(np.ceil(draws / 2)):draws, dam-vm+1:dam+2], axis=0)))
                    # betavs2 error in std deviation formatting
                betavs3 = np.array(range(dam-vm+2, dam+2))
                betavs = np.transpose(np.array([betavs,betavs2, betavs3]))
                if np.shape(betavs)[1] > 0:
                    sortInds = np.argsort(betavs[:, 0])
                    betavs = betavs[sortInds]
                killset = []
                evmin = ev
                for i in range(0, vm):
                    if betavs[i, 1] > threshstdb or betavs[i, 1] > threshstda and betavs[i, 0] < threshav * \
                            np.mean(np.abs(np.mean(beters[int(np.ceil(draws/2)):draws, 0]))):  # index to 'beters'
                        killtest = np.append(killset, (betavs[i, 2] - 1))
                        if killtest.size > 1:
                            killtest[::-1].sort()  # max to min so damtx_test rows get deleted in order of end to start
                        damtx_test = damtx
                        for k in range(0, np.size(killtest)):
                            damtx_test = np.delete(damtx_test, int(np.array(killtest[k])-1), 0)
                        damtest, null = np.shape(damtx_test)
                        [betertest, null, null, null, Xtest, evtest] = sampler_fn(inputs, data, phis, X, damtx_test, a, b,
                                                                             atau, btau, draws, phind, xsm, sigsqd0,
                                                                             tausqd0, dtd)
                        if aic:
                            evtest = evtest + (2 - np.log(n))*(damtest+1)
                        if evtest < evmin:
                            killset = killtest
                            evmin = evtest
                            xers = Xtest
                            beters = betertest
                for k in range(0, np.size(killset)):
                    damtx = np.delete(damtx, int(np.array(killset[k]) - 1), 0)
                ev = evmin
                X = xers
                if self.ConsoleOutput:
                    if data.dtype != np.float64:  # if large dataset, then 'Gibbs: 100.00%' printed from inside gibbs
                        sys.stdout.write('\r')  # place cursor at start of line to erase 'Gibbs: 100.00%'
                    print([ind, ev])
                if np.size(evs) > 0:
                    if ev < np.min(evs):
                        betas = beters
                        mtx = damtx
                        greater = 1
                        evs = np.append(evs, ev)
                    elif greater < tolerance:
                        greater = greater + 1
                        evs = np.append(evs, ev)
                    else:
                        finished = 1
                        evs = np.append(evs, ev)
                        break
                else:
                    greater = greater + 1
                    betas = beters
                    mtx = damtx
                    evs = np.append(evs, ev)
                if m == 1:
                    break
                elif way3:
                    if indvec[1] > indvec[2]:
                        indvec[0] = indvec[0] + 1
                        indvec[1] = indvec[1] - 1
                    elif indvec[2]:
                        indvec[1] = indvec[1] + 1
                        indvec[2] = indvec[2] - 1
                        if indvec[1] > indvec[0]:
                            indvec[0] = indvec[0] + 1
                            indvec[1] = indvec[1] - 1
                    else:
                        break
                elif indvec[1]:
                    indvec[0] = indvec[0] + 1
                    indvec[1] = indvec[1] - 1
                else:
                    break
            if finished != 0:
                break
            ind = ind + 1
            if ind > len(phis):
                break
        # Implementation of 'gimme' feature
        if gimmie:
            betas = beters
            mtx = damtx
        self.betas = betas[-self.draws::, :]  # discard 'burnin' draws by only keeping last 'draws' draws
        self.mtx = mtx
        self.evs = evs
        return betas[-self.draws::, :], mtx, evs  # discard 'burnin'



# Dictionary to map sampler names to classes
SAMPLERS_DICT = {
    "gibbs": gibbs,
    "gibbs_update": gibbs_Xin_update
}
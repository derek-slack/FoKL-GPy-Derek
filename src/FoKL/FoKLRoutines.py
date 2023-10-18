from FoKL import getKernels # from FoKL import getKernels
import pandas as pd
import warnings
import itertools
import math
import numpy as np
import pdb
from numpy import linalg as LA
from scipy.linalg import eigh
import matplotlib.pyplot as plt

class FoKL:
    def __init__(self, **kwargs):
        """
            initialization inputs:
        
                - 'phis' is a data structure with the spline coefficients for the basis
                functions, built with 'spline_coefficient.txt' and 'splineconvert' or
                'spline_coefficient_500.txt' and 'splineconvert500' (the former provides
                25 basis functions: enough for most things -- while the latter provides
                500: definitely enough for anything)
    
                - 'relats_in' is a boolean matrix indicating which terms should be excluded
                from the model building; for instance if a certain main effect should be
                excluded relats will include a row with a 1 in the column for that input
                and zeros elsewhere; if a certain two way interaction should be excluded
                there should be a row with ones in those columns and zeros elsewhere.
                to exclude no terms 'relats = np.array([[0]])'. an example of excluding
                the first input main effect and its interaction with the third input for
                a case with three total inputs is: 'relats = np.array([[1,0,0],[1,0,1]])'
    
                - 'a' and 'b' are the shape and scale parameters of the ig distribution for
                the observation error variance of the data. the observation error model is
                white noise. choose the mode of the ig distribution to match the noise in
                the output dataset and the mean to broaden it some
    
                - 'atau' and 'btau' are the parameters of the ig distribution for the 'tau
                squared' parameter: the variance of the beta priors is iid normal mean
                zero with variance equal to sigma squared times tau squared. tau squared
                must be scaled in the prior such that the product of tau squared and sigma
                squared scales with the output dataset       
    
                - 'tolerance' controls how hard the function builder tries to find a better
                model once adding terms starts to show diminishing returns. a good
                default is 3 -- large datasets could benefit from higher values
    
                - 'draws' is the total number of draws from the posterior for each tested
                model
    
                - 'gimmie' is a boolean causing the routine to return the most complex
                model tried instead of the model with the optimum bic
    
                - 'way3' is a boolean specifying the calculation of three-way interactions
            
                - 'threshav' and 'threshstda' form a threshold for the elimination of terms
                    - 'threshav' is a threshold for proposing terms for elimination based on
                    their mean values (larger thresholds lead to more elimination)
                    - 'threshstda' is a threshold standard deviation -- expressed as a fraction 
                    relative to the mean
                    - terms with coefficients that are lower than 'threshav' and higher than
                    'threshstda' will be proposed for elimination (elimination will happen or not 
                    based on relative BIC values)
    
                - 'threshstdb' is a threshold standard deviation that is independent of the
                mean value of the coefficient -- all with a standard deviation (fraction 
                relative to mean) exceeding this value will be proposed for elimination
    
                - 'aic' is a boolean specifying the use of the aikaike information
                criterion

            default values:
    
                - phis = getKernels.sp500()
                - relats_in = [] or np.array([[0]]) ??? = [], [1,1,1,1,1,1]
                - a = 4 ??? 9, 1000
                - b = 0.1 ??? 0.01, 1
                - atau = std dev of ??? 3, 4
                - btau = std dev of ??? 4000, 0.6091
                - tolerance = 3
                - draws = 1000 ??? 1000, 2000
                - gimmie = False
                - way3 = False
                - threshav = 0.05 ??? 0.05, 0
                - threshstda = 0.5 ??? 0.5, 0
                - threshstdb = 2 ??? 2, 100
                - aic = False
        """
    
        # Calculate some default hypers based on data unless user-defined:
        if 'btau' not in kwargs:
            btau = 1000 # stdev(inputs) # NEEDS TO BE UPDATED WITH CORRECT EQUATION
            # . . . (or updated in model.fit since data not provided here)
            # . . . (or create function for user to call before initializing FoKLRoutines so as to provide btau)
        else:
            btau = kwargs.get('btau')

        # Define default hypers:
        hypers = {'phis': getKernels.sp500(),'relats_in': [],'a': 4,'b': 0.01,'atau': 4,'btau': btau,'tolerance': 3,'draws': 1000,'gimmie': False,'way3': False,'threshav': 0.05,'threshstda': 0.5,'threshstdb': 2,'aic': False}

        # Update hypers based on user-input:
        kwargs_expected = hypers.keys()
        for kwarg in kwargs.keys():
            if kwarg not in kwargs_expected:
                raise ValueError(f"Unexpected keyword argument: {kwarg}")
            else:
                hypers[kwarg] = kwargs.get(kwarg, hypers.get(kwarg))
        for hyperKey, hyperValue in hypers.items():
            setattr(self, hyperKey, hyperValue) # defines each hyper as an attribute of 'self'
            locals()[hyperKey] = hyperValue # defines each hyper as a local variable

    def splineconvert500(self,A):
        """
        Same as splineconvert, but for a larger basis of 500
        """

        coef = np.loadtxt(A)

        phi = []
        for i in range(500):
            a = coef[i * 499:(i + 1) * 499, 0]
            b = coef[i * 499:(i + 1) * 499, 1]
            c = coef[i * 499:(i + 1) * 499, 2]
            d = coef[i * 499:(i + 1) * 499, 3]

            phi.append([a, b, c, d])

        return phi

    def coverage3(self, inputs, data, draws, plots):
        """
            Inputs:
                Interprets outputs of FoKL.fit()

                betas - betas emulator output

                inputs - normalized inputs

                draws - number of beta terms used

                plots - binary for plot output

            returns:
                Meen: Predicted values for each indexed input

                RSME: root mean squared deviation

                Bounds: confidence interval, dotted lines on plot, larger bounds means more uncertainty at location


           """
        betas = self.betas
        mtx = self.mtx
        phis = self.phis

        m, mbets = np.shape(betas)  # Size of betas
        n, mputs = np.shape(inputs)  # Size of normalized inputs

        setnos_p = np.random.randint(m, size=(1, draws))  # Random draws  from integer distribution
        i = 1
        while i == 1:
            setnos = np.unique(setnos_p)

            if np.size(setnos) == np.size(setnos_p):
                i = 0
            else:
                setnos_p = np.append(setnos, np.random.randint(m, size=(1, draws - np.shape(setnos)[0])))

        X = np.zeros((n, mbets))
        inputs = np.asarray(inputs)
        for i in range(n):
            phind = []  # Rounded down point of input from 0-499
            for j in range(len(inputs[i])):
                phind.append(math.floor(inputs[i, j] * 498))
                # 499 changed to 498 for python indexing

            phind_logic = []
            for k in range(len(phind)):
                if phind[k] == 498:
                    phind_logic.append(1)
                else:
                    phind_logic.append(0)

            phind = np.subtract(phind, phind_logic)

            for j in range(1, mbets):
                phi = 1
                for k in range(mputs):
                    num = mtx[j - 1, k]
                    if num > 0:
                        xsm = 498 * inputs[i][k] - phind[k]
                        phi = phi * (phis[int(num) - 1][0][phind[k]] + phis[int(num) - 1][1][phind[k]] * xsm +
                                     phis[int(num) - 1][2][phind[k]] * xsm ** 2 + phis[int(num) - 1][3][
                                         phind[k]] * xsm ** 3)
                X[i, j] = phi

        X[:, 0] = np.ones((n,))
        modells = np.zeros((np.shape(data)[0], draws))
        for i in range(draws):
            modells[:, i] = np.matmul(X, betas[setnos[i], :])
        meen = np.mean(modells, 1)
        bounds = np.zeros((np.shape(data)[0], 2))
        cut = int(np.floor(draws * .025))
        for i in range(np.shape(data)[0]):
            drawset = np.sort(modells[i, :])
            bounds[i, 0] = drawset[cut]
            bounds[i, 1] = drawset[draws - cut]

        if plots:
            plt.plot(meen, 'b', linewidth=2)
            plt.plot(bounds[:, 0], 'k--')
            plt.plot(bounds[:, 1], 'k--')

            plt.plot(data, 'ro')

            plt.show()

        rmse = np.sqrt(np.mean(meen - data) ** 2)
        return meen, bounds, rmse

    def fit(self, inputs, data, p_true, **kwargs):
        """
            inputs: 
                'inputs' - normalzied inputs

                'data' - results

                'p_true' - (optional) percentage 0 to 1 of datapoints to use for training the model.
                set equal to 1 (default) or leave blank to disable the auto split and fit the model to all data.

            outputs:
                 'betas' are a draw from the posterior distribution of coefficients: matrix, with
                 rows corresponding to draws and columns corresponding to terms in the GP

                 'mtx' is the basis function interaction matrix from the
                 best model: matrix, with rows corresponding to terms in the GP (and thus to the 
                 columns of 'betas' and columns corresponding to inputs. a given entry in the 
                 matrix gives the order of the basis function appearing in a given term in the GP.
                 all basis functions indicated on a given row are multiplied together.
                 a zero indicates no basis function from a given input is present in a given term

                 'ev' is a vector of BIC values from all of the models
                 evaluated

             attributes:
                'inputs_train', 'data_train', 'inputs_test', and 'data_test' get stored as attributes of 'self'.
                if 'p_true' = 1, then all datapoints fall under the train set and the test set will be an empty list [].
        """

        if p_true in kwargs:
            p_true = kwargs.get(p_true)
        else:
            p_true = 1

        # Ensure p_true is a scalar value
        p_true = float(p_true) if isinstance(p_true, pd.Series) else p_true

        # Automatically handle some data formatting exceptions:
        def auto_cleanData(inputs, data, p_true):
            
            # automatically appends input values into a form that can be used in the fit function.
            for i in range(len(inputs)):
                inputs.append(inputs.iloc[i])
            
            # Convert 'inputs' and 'datas' to numpy if pandas:
            if isinstance(inputs, pd.DataFrame):
                inputs = inputs.to_numpy()
                warnings.warn("Warning: 'inputs' was auto-converted to numpy. Convert manually for assured accuracy.", UserWarning)
            if isinstance(data, pd.DataFrame):
                data = data.to_numpy()
                warnings.warn("Warning: 'data' was auto-converted to numpy. Convert manually for assured accuracy.", UserWarning)
                
            # Normalize 'inputs' and convert to proper format for FoKL:
            inputs = np.array(inputs) # attempts to handle lists or any other format (i.e., not pandas)
            inputs = np.atleast_2d(inputs)
            # . . . inputs = {ndarray: (N, M)} = {ndarray: (datapoints, input variables)} =
            # . . . . . . array([[x1(t1),x2(t1),...,xM(t1)],[x1(t2),x2(t2),...,xM(t2)],...,[x1(tN),x2(tN),...,xM(tN)]])
            N = inputs.shape[0]
            M = inputs.shape[1]
            if M > N: # if more "input variables" than "datapoints", assume user is using transpose of proper format above
                inputs = inputs.transpose()
                warnings.warn("Warning: 'inputs' was transposed. Ignore if more datapoints than input variables.", category=UserWarning)
            inputs_max = np.max(inputs, axis=0) # max of each input variable
            inputs_scale = []
            for ii in range(len(inputs_max)):
                inputs_min = np.min(inputs[:, ii])
                if inputs_max[ii] != 1 or inputs_min != 0:
                    if inputs_min == inputs_max[ii]:
                        inputs[:,ii] = np.ones(len(inputs[:,ii]))
                        warnings.warn("Warning: 'inputs' contains a column of constants which will not improve the model's fit.", category=UserWarning)
                    else: # normalize
                        inputs[:,ii] = (inputs[:,ii] - inputs_min) / (inputs_max[ii] - inputs_min)
                inputs_scale.append(np.array([inputs_min, inputs_max[ii]]))  # store for post-processing convenience
            inputs = inputs.tolist() # convert to list, which is proper format for FoKL, like:
            # . . . {list: N} = [[x1(t1),x2(t1),...,xM(t1)],[x1(t2),x2(t2),...,xM(t2)],...,[x1(tN),x2(tN),...,xM(tN)]]
            # Transpose 'data' if needed:
            data = np.array([data])  # attempts to handle lists or any other format (i.e., not pandas)
            if data.ndim == 1:  # if data.shape == (number,) != (number,1), then add new axis to match FoKL format
                data = data[:, np.newaxis]
                warnings.warn("Warning: 'data' was made into (n,1) column vector from single list (n,) to match FoKL formatting.",category=UserWarning)
            else: # check user provided only one output column/row, then transpose if needed
                N = data.shape[0]
                M = data.shape[1]
                if (M != 1 and N != 1) or (M == 1 and N == 1):
                    raise ValueError("Error: 'data' must be a vector.")
                elif M != 1 and N == 1:
                    data = data.transpose()
                    warnings.warn("Warning: 'data' was transposed to match FoKL formatting.",category=UserWarning)

            if p_true < 1:
                def random_train(p_true, inputs, data):  # split data for training and testing (i.e., validating)
                    Ldata = len(data)
                    train_log = np.random.rand(Ldata) < p_true # indices to use as training data
                    test_log = ~train_log

                    inputs_train = [inputs[ii] for ii, ele in enumerate(train_log) if ele] # because list
                    data_train = data[train_log] # because numpy
                    inputs_test = [inputs[ii] for ii, ele in enumerate(test_log) if ele]
                    data_test = data[test_log]

                    return inputs_train, data_train, inputs_test, data_test, p_true
                inputs_train, data_train, inputs_test, data_test, p_true = random_train(p_true, inputs, data)
            else:
                inputs_train = inputs
                data_train = data
                inputs_test = []
                data_test = []
            return inputs_train, data_train, inputs_test, data_test, inputs_scale, p_true


        inputs, data, inputs_validn, data_validn, inputs_scale, p_true = auto_cleanData(inputs, data, p_true)
        
        self.inputs = inputs
        self.data = data
        self.inputs_validn = inputs_validn
        self.data_validn = data_validn
        self.inputs_scale = inputs_scale # [min,max] of each input before normalization

        def inputs_to_numpy(inputs_list):
            inputs_np = np.array(inputs_list) # should be N datapoints x M inputs
            NM = np.shape(inputs_np)
            if NM[0] < NM[1]:
                inputs_np = np.transpose(inputs_np)
            return inputs_np
        inputs = inputs_to_numpy(self.inputs)
        
        # Initializations:
        phis = self.phis
        relats_in = self.relats_in
        a = self.a
        b = self.b
        atau = self.atau
        btau = self.btau
        tolerance = self.tolerance
        draws = self.draws
        gimmie = self.gimmie
        way3 = self.way3
        threshav = self.threshav
        threshstda = self.threshstda
        threshstdb = self.threshstdb
        aic = self.aic

        def perms(x):
            """Python equivalent of MATLAB perms."""
            # from https://stackoverflow.com/questions/38130008/python-equivalent-for-matlabs-perms
            a = np.vstack(list(itertools.permutations(x)))[::-1]

            return a
        def gibbs(inputs, data, phis, Xin, discmtx, a, b, atau, btau, draws):
            """

            'inputs' is the set of normalized inputs -- both parameters and model
            inputs -- with columns corresponding to inputs and rows the different
            experimental designs

            'data' are the experimental results: column vector, with entries
            corresponding to rows of 'inputs'

            'phis' are a data structure with the spline coefficients for the basis
            functions, built with 'BasisSpline.txt' and 'splineloader' or
            'splineconvert'

            'discmtx' is the interaction matrix for the bss-anova function -- rows
            are terms in the function and columns are inputs (cols should line up
            with cols in 'inputs'

            'a' and 'b' are the parameters of the ig distribution for the
            observation error variance of the data

            'atau' and 'btau' are the parameters of the ig distribution for the 'tau
            squared' parameter: the variance of the beta priors

            'draws' is the total number of draws
            """
            # building the matrix by calculating the corresponding basis function
            # outputs for each set of inputs
            minp, ninp = np.shape(inputs)
            phi_vec = []
            # CHECK BELOW !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            if np.shape(discmtx) == ():  # part of fix for single input model
                mmtx = 1
            else:
                mmtx, null = np.shape(discmtx)

            if np.size(Xin) == 0:
                Xin = np.ones((minp, 1))
                mxin, nxin = np.shape(Xin)
            else:
                # X = Xin
                mxin, nxin = np.shape(Xin)
            if mmtx - nxin < 0:
                X = Xin
            else:
                X = np.append(Xin, np.zeros((minp, mmtx - nxin)), axis=1)
            # CHECK ABOVE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            for i in range(minp):

                phind = []
                for j in range(len(inputs[i])):
                    phind.append(math.ceil(inputs[i][j] * 498))
                    # 499 changed to 498 for python indexing

                phind_logic = []
                for k in range(len(phind)):
                    if phind[k] == 0:
                        phind_logic.append(1)
                    else:
                        phind_logic.append(0)

                phind = np.add(phind, phind_logic)

                for j in range(nxin, mmtx + 1):
                    null, nxin2 = np.shape(X)
                    if j == nxin2:
                        X = np.append(X, np.zeros((minp, 1)), axis=1)

                    phi = 1

                    for k in range(ninp):

                        if np.shape(discmtx) == ():
                            num = discmtx
                        else:
                            num = discmtx[j - 1][k]

                        if num != 0:  # enter if loop if num is nonzero
                            xsm = 498 * inputs[i][k] - phind[k]
                            phi = phi * (phis[int(num) - 1][0][phind[k]] + phis[int(num) - 1][1][phind[k]] * xsm +
                                         phis[int(num) - 1][2][phind[k]] * xsm ** 2 + phis[int(num) - 1][3][phind[k]] *
                                         xsm ** 3)

                    X[i][j] = phi

            # initialize tausqd at the mode of its prior: the inverse of the mode of
            # sigma squared, such that the initial variance for the betas is 1
            sigsqd = b / (1 + a)
            tausqd = btau / (1 + atau)

            XtX = np.transpose(X).dot(X)

            Xty = np.transpose(X).dot(data)

            # See the link https://stackoverflow.com/questions/8765310/scipy-linalg-eig-return-complex-eigenvalues-for-covariance-matrix
            Lamb, Q = eigh(XtX)  # using scipy eigh function to avoid generation of imaginary values due to numerical errors
            # Lamb, Q = LA.eig(XtX)

            Lamb_inv = np.diag(1 / Lamb)

            betahat = Q.dot(Lamb_inv).dot(np.transpose(Q)).dot(Xty)
            squerr = LA.norm(data - X.dot(betahat)) ** 2

            astar = a + 1 + len(data) / 2 + (mmtx + 1) / 2
            atau_star = atau + mmtx / 2

            dtd = np.transpose(data).dot(data)

            # Gibbs iterations

            betas = np.zeros((draws, mmtx + 1))
            sigs = np.zeros((draws, 1))
            taus = np.zeros((draws, 1))

            lik = np.zeros((draws, 1))
            n = len(data)

            for k in range(draws):

                Lamb_tausqd = np.diag(Lamb) + (1 / tausqd) * np.identity(mmtx + 1)
                Lamb_tausqd_inv = np.diag(1 / np.diag(Lamb_tausqd))

                mun = Q.dot(Lamb_tausqd_inv).dot(np.transpose(Q)).dot(Xty)
                if mun.ndim == 1: # if mun.shape == (number,) != (number,1), then add new axis
                    mun = mun[:, np.newaxis]
                    warnings.warn("Warning: 'mun' was made into (n,1) column vector from single list (n,). It is unclear why this was not already the case.",category=UserWarning)
                S = Q.dot(np.diag(np.diag(Lamb_tausqd_inv) ** (1 / 2)))

                vec = np.random.normal(loc=0, scale=1, size=(mmtx + 1, 1))  # drawing from normal distribution
                betas[k][:] = np.transpose(mun + sigsqd ** (1 / 2) * (S).dot(vec))

                vecc = mun - np.reshape(betas[k][:], (len(betas[k][:]), 1))


                bstar = b + 0.5 * (
                            betas[k][:].dot(XtX.dot(np.transpose([betas[k][:]]))) - 2 * betas[k][:].dot(Xty) + dtd + betas[
                                                                                                                         k][
                                                                                                                     :].dot(
                        np.transpose([betas[k][:]])) / tausqd)
                # bstar = b + comp1.dot(comp2) + 0.5 * dtd - comp3;

                # Returning a 'not a number' constant if bstar is negative, which would
                # cause np.random.gamma to return a ValueError
                if bstar < 0:
                    sigsqd = math.nan
                else:
                    sigsqd = 1 / np.random.gamma(astar, 1 / bstar)

                sigs[k] = sigsqd

                btau_star = (1 / (2 * sigsqd)) * (betas[k][:].dot(np.reshape(betas[k][:], (len(betas[k][:]), 1)))) + btau

                tausqd = 1 / np.random.gamma(atau_star, 1 / btau_star)
                taus[k] = tausqd

                # Calculate the evidence
            siglik = np.var(data - np.matmul(X, betahat))

            lik = -(n / 2) * np.log(siglik) - (n - 1) / 2
            ev = (mmtx + 1) * np.log(n) - 2 * np.max(lik)

            X = X[:, 0:mmtx + 1]

            return betas, sigs, taus, betahat, X, ev



        # 'n' is the number of datapoints whereas 'm' is the number of inputs
        n, m = np.shape(inputs)
        mrel = n
        damtx = np.array([])
        evs = np.array([])

        # Conversion of Lines 79-100 of emulator_Xin.m
        if np.logical_not(all([isinstance(index, int) for index in relats_in])):  # checks relats to see if it's an array
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

            while 1:
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
                [dam,null] = np.shape(damtx)

                [beters, null, null, null, xers, ev] = gibbs(inputs, data, phis, X, damtx, a, b, atau, btau, draws)

                if aic:
                    ev = ev + (2 - np.log(n)) * (dam + 1)

                betavs = np.abs(np.mean(beters[int(np.ceil((draws / 2)+1)):draws, (dam - vm + 1):dam+1], axis=0))
                betavs2 = np.divide(np.std(np.array(beters[int(np.ceil(draws/2)+1):draws, dam-vm+1:dam+1]), axis=0), np.abs(np.mean(beters[int(np.ceil(draws / 2)):draws, dam-vm+1:dam+2], axis=0))) # betavs2 error in std deviation formatting
                betavs3 = np.array(range(dam-vm+2, dam+2))
                betavs = np.transpose(np.array([betavs,betavs2, betavs3]))
                if np.shape(betavs)[1] > 0:
                    sortInds = np.argsort(betavs[:, 0])
                    betavs = betavs[sortInds]

                killset = []
                evmin = ev



                for i in range(0, vm):


                    if betavs[i, 1] > threshstdb or betavs[i, 1] > threshstda and betavs[i, 0] < threshav * np.mean(np.abs(np.mean(beters[int(np.ceil(draws/2 +1)):draws, 0]))):

                        killtest = np.append(killset, (betavs[i, 2] - 1))
                        damtx_test = damtx
                        count = 1
                        for k in range(0, np.size(killtest)):
                            damtx_test = np.delete(damtx_test, (int(np.array(killtest[k]))-count), 0)
                            count = count + 1
                        damtest, null = np.shape(damtx_test)

                        [betertest, null, null, null, Xtest, evtest] = gibbs(inputs, data, phis, X, damtx_test, a, b, atau, btau, draws)
                        if aic:
                            evtest = evtest + (2 - np.log(n))*(damtest+1)
                        if evtest < evmin:
                            killset = killtest
                            evmin = evtest
                            xers = Xtest
                            beters = betertest
                count = 1
                for k in range(0, np.size(killset)):
                    damtx = np.delete(damtx, (int(np.array(killset[k])) - count), 0)
                    count = count + 1

                ev = evmin
                X = xers

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

        self.betas = betas
        self.mtx = mtx
        self.evs = evs

        return betas, mtx, evs

    def GP_Integrate(self, betas, matrix, b, norms, phis, start, stop, y0, h, used_inputs):
        """""
          betas is a list of arrays in which each entry to the list contains a specific row of the betas matrix,
          or the mean of the the betas matrix for each model being integrated

          matrix is a list of arrays containing the interaction matrix of each model

          b is an array of of the values of all the other inputs to the model(s) (including
          any forcing functions) over the time period we integrate over. The length of b
          should be equal to the number of points in the final time series (end-start)/h
          All values in b need to be normalized with respect to the min and max values
          of their respective values in the training dataset

          h is the step size with respect to time

          norms is a matrix of the min and max values of all the inputs being
          integrated (in the same order as y0). min values are in the top row, max values in the bottom.

          Start is the time at which integration begins. Stop is the time to
          end integration.

          y0 is an array of the inital conditions for the models being integrated

          Used inputs is a list of arrays containing the information as to what inputs
          are used in what model. Each array should contain a vector corresponding to a different model.
          Inputs should be referred to as those being integrated first, followed by
          those contained in b (in the same order as they appear in y0 and b
          respectively)
          For example, if two models were being integrated, with 3 other inputs total
          and the 1st model used both models outputs as inputs and the 1st and 3rd additional
          inputs, while the 2nd model used its own output as an input and the 2nd
          and 3rd additional inputs, used_inputs would be equal to
          [[1,1,1,0,1],[0,1,0,1,0]].
          If the models created do not follow this ordering scheme for their inputs
          the inputs can be rearranged based upon an alternate
          numbering scheme provided to used_inputs. E.g. if the inputs need to breordered the the 1st input should have a '1' in its place in the
          used_inputs vector, the 2nd input should have a '2' and so on. Using the
          same example as before, if the 1st models inputs needed rearranged so that
          the 3rd additional input came first, followed by the two model outputs in
          the same order as they are in y0, and ends with the 1st additional input,
          then the 1st cell in used_inputs would have the form [2,3,4,0,1]

          T an array of the time steps the models are integrated at.

          Y is an array of the models that have been integrated, at the time steps
          contained in T.
          """

        def prediction(inputs):
            f = []
            for kk in range(len(inputs)):
                if len(f) == 0:
                    f = [bss_eval(inputs[kk], betas[kk], phis, matrix[kk])]
                else:
                    f = np.append(f, bss_eval(inputs[kk], betas[kk], phis, matrix[kk]))
            return f

        def reorder(used, inputs):
            order = used[used != 0]
            reinputs = np.array((inputs.shape))
            for i in range(len(inputs)):
                reinputs[order[i] - 1] = inputs[i]
            return reinputs

        def normalize(v, minim, maxim):
            norm = np.zeros((1, 1))
            norm[0] = (v - minim) / (maxim - minim)
            if norm[0] > 1:
                norm[0] = 1
            if norm[0] < 0:
                norm[0] = 0
            return norm

        def bss_eval(x, betas, phis, mtx, Xin=[]):
            """
            x are normalized inputs
            betas are coefficients. If using 'Xin' include all betas (include constant beta)

            phis are the spline coefficients for the basis functions (cell array)

            mtx is the 'interaction matrix' -- a matrix each row of which corresponds
            to a term in the expansion, each column corresponds to an input. if the
            column is zero there's no corresponding basis function in the term; if
            it's greater than zero it corresponds to the order of the basis function

            Xin is an optional input of the chi matrix. If this was pre-computed with xBuild,
            one may use it to improve performance.
            """

            if Xin == []:
                m, n = np.shape(mtx)  # getting dimensions of the matrix 'mtx'

                mx = 1

                mbet = 1

                delta = np.zeros([mx, mbet])

                phind = []

                for j in range(len(x)):
                    phind.append(math.floor(x[j] * 498))

                phind_logic = []
                for k in range(len(phind)):
                    if phind[k] == 498:
                        phind_logic.append(1)
                    else:
                        phind_logic.append(0)

                phind = np.subtract(phind, phind_logic)

                r = 1 / 498
                xmin = r * np.array(phind)
                X = (x - xmin) / r
                for ii in range(mx):
                    for i in range(m):
                        phi = 1

                        for j in range(n):

                            num = mtx[i][j]

                            if num != 0:
                                phi = phi * (phis[int(num) - 1][0][phind[j]] + phis[int(num) - 1][1][phind[j]] * X[j] \
                                             + phis[int(num) - 1][2][phind[j]] * X[j] ** 2 + phis[int(num) - 1][3][
                                                 phind[j]] *
                                             X[j] ** 3)

                        delta[ii, :] = delta[ii, :] + betas[i + 1] * phi
                        mmm = 1
                delta[ii, :] = delta[ii, :] + betas[0]
            else:
                if np.ndim(betas) == 1:
                    betas = np.array([betas])
                elif np.ndim(betas) > 2:
                    print("The \'betas\' parameter has %d dimensions, but needs to have only 2." % (np.ndim(betas)))
                    print("The current shape is:", np.shape(betas))
                    print("Attempting to get rid of unnecessary dimensions of size 1...")
                    betas = np.squeeze(betas)

                    if np.ndim(betas) == 1:
                        betas = np.array([betas])
                        print("Success! New shape is", np.shape(betas))
                    elif np.ndim(betas) == 2:
                        print("Success! New shape is", np.shape(betas))

                delta = Xin.dot(betas.T)
            dc = 1
            return delta

        T = np.arange(start, stop + h, h)
        y = y0
        Y = np.array([y0])
        Y = Y.reshape(len(y0), 1)

        ind = 1
        for t in range(len(T) - 1):
            inputs1 = list()
            othinputs = list()
            inputs2 = list()
            inputs3 = list()
            inputs4 = list()
            for i in range(len(y)):  # initialize inputs1 and othinputs to contain empty arrays
                inputs1.append([])
                othinputs.append([])
                inputs2.append([])
                inputs3.append([])
                inputs4.append([])
            for i in range(len(y)):
                for j in range(len(y)):
                    if used_inputs[i][j] != 0:
                        if len(inputs1[i]) == 0:
                            inputs1[i] = normalize(y[j], norms[0, j], norms[1, j])
                        else:
                            inputs1[i] = np.append(inputs1[i], normalize(y[j], norms[0, j], norms[1, j]), 1)

            nnn = int(b.size / b.shape[0])
            if b.size > 0:
                for ii in range(len(y0)):
                    for jj in range(len(y), nnn + len(y)):

                        if used_inputs[ii][jj] != 0:
                            if len(othinputs[ii]) == 0:
                                if ind - 1 == 0:
                                    othinputs[ii] = b[ind - 1]
                                else:

                                    othinputs[ii] = b[ind - 1]

                                    ttt = 1
                            else:
                                othinputs[ii] = np.append(othinputs[ii], b[ind - 1, jj - len(y0)], 1)
                for k in range(len(y)):
                    inputs1[k] = np.append(inputs1[k], othinputs[k])
            for ii in range(len(y0)):
                if np.amax(used_inputs[ii]) > 1:
                    inputs1[ii] = reorder(used_inputs[ii], inputs1[ii])

            dy1 = prediction(inputs1) * h

            for p in range(len(y)):
                if y[p] >= norms[1, p] and dy1[p] > 0:
                    dy1[p] = 0
                else:
                    if y[p] <= norms[0, p] and dy1[p] < 0:
                        dy1[p] = 0

            for i in range(len(y)):
                for j in range(len(y)):
                    if used_inputs[i][j] != 0:
                        if len(inputs2[i]) == 0:
                            inputs2[i] = normalize(y[j] + dy1[j] / 2, norms[0, j], norms[1, j])
                        else:
                            inputs2[i] = np.append(inputs2[i], normalize(y[j] + dy1[j] / 2, norms[0, j], norms[1, j]),
                                                   1)

            for k in range(len(y)):
                inputs2[k] = np.append(inputs2[k], othinputs[k])
            for ii in range(len(y0)):
                if np.amax(used_inputs[ii]) > 1:
                    inputs2[ii] = reorder(used_inputs[ii], inputs2[ii])
            dy2 = prediction(inputs2) * h
            for p in range(len(y)):
                if (y[p] + dy1[p] / 2) >= norms[1, p] and dy2[p] > 0:
                    dy2[p] = 0
                if (y[p] + dy1[p] / 2) <= norms[0, p] and dy2[p] < 0:
                    dy2[p] = 0

            for i in range(len(y)):
                for j in range(len(y)):
                    if used_inputs[i][j] != 0:
                        if len(inputs3[i]) == 0:
                            inputs3[i] = normalize(y[j] + dy2[j] / 2, norms[0, j], norms[1, j])
                        else:
                            inputs3[i] = np.append(inputs3[i], normalize(y[j] + dy2[j] / 2, norms[0, j], norms[1, j]),
                                                   1)
            if b.size > 0:
                for k in range(len(y)):
                    inputs3[k] = np.append(inputs3[k], othinputs[k])
            for ii in range(len(y0)):
                if np.amax(used_inputs[ii]) > 1:
                    inputs3[ii] = reorder(used_inputs[ii], inputs3[ii])
            dy3 = prediction(inputs3) * h
            for p in range(len(y)):
                if (y[p] + dy2[p] / 2) >= norms[1, p] and dy3[p] > 0:
                    dy3[p] = 0
                if (y[p] + dy2[p] / 2) <= norms[0, p] and dy3[p] < 0:
                    dy3[p] = 0

            for i in range(len(y)):
                for j in range(len(y)):
                    if used_inputs[i][j] != 0:
                        if len(inputs4[i]) == 0:
                            inputs4[i] = normalize(y[j] + dy3[j], norms[0, j], norms[1, j])
                        else:
                            inputs4[i] = np.append(inputs4[i], normalize(y[j] + dy3[j], norms[0, j], norms[1, j]), 1)
            if b.size > 0:
                for k in range(len(y)):
                    inputs4[k] = np.append(inputs4[k], othinputs[k])
            for ii in range(len(y0)):
                if np.amax(used_inputs[ii]) > 1:
                    inputs4[ii] = reorder(used_inputs[ii], inputs4[ii])
            dy4 = prediction(inputs4) * h
            for p in range(len(y)):
                if (y[p] + dy3[p]) >= norms[1, p] and dy4[p] > 0:
                    dy4[p] = 0
                if (y[p] + dy3[p]) <= norms[0, p] and dy4[p] < 0:
                    dy4[p] = 0

            y += (dy1 + 2 * dy2 + 2 * dy3 + dy4) / 6
            yt = np.reshape(y, [2, 1])
            Y = np.append(Y, yt, 1)
            ind += 1

        return T, Y



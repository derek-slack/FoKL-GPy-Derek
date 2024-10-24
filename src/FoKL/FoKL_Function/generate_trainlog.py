def generate_trainlog(self, train, n=None):
        """Generate random logical vector of length 'n' with 'train' percent as True."""
        if train < 1:
            if n is None:
                n = self.inputs.shape[0]  # number of observations
            l_log = int(n * train)  # required length of indices for training
            if l_log < 2:
                l_log = int(2)  # minimum required for training set
            trainlog_i = np.array([], dtype=int)  # indices of training data (as an index)
            while len(trainlog_i) < l_log:
                trainlog_i = np.append(trainlog_i, np.random.random_integers(n, size=l_log) - 1)
                trainlog_i = np.unique(trainlog_i)  # also sorts low to high
                np.random.shuffle(trainlog_i)  # randomize sort
            if len(trainlog_i) > l_log:
                trainlog_i = trainlog_i[0:l_log]  # cut-off extra indices (beyond 'percent')
            trainlog = np.zeros(n, dtype=bool)  # indices of training data (as a logical)
            for i in trainlog_i:
                trainlog[i] = True
        else:
            # trainlog = np.ones(n, dtype=bool)  # wastes memory, so use following method coupled with 'trainset':
            trainlog = None  # means use all observations
        return trainlog


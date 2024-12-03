class Trainset:
    def __init__(self, inputs, data, trainset):
        self.trainset = trainset
        self.inputs = inputs
        self.data = data
    def trainset(self):
        """
        After running 'clean', call 'trainset' to get train inputs and train data. The purpose of this method is to
        simplify syntax, such that the code here does not need to be re-written each time the train set is defined.
        traininputs, traindata = self.trainset()
        """
        if self.trainlog is None:  # then use all observations for training
            return self.inputs, self.data
        else:  # self.trainlog is vector indexing observations
            return self.inputs[self.trainlog, :], self.data[self.trainlog]
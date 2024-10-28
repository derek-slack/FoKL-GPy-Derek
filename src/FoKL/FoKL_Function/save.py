def save(self, filename=None, directory=None):
    """
    Save a FoKL class as a file. By default, the 'filename' is 'model_yyyymmddhhmmss.fokl' and is saved to the
    directory of the Python script calling this method. Use 'directory' to change the directory saved to, or simply
    embed the directory manually within 'filename'.
    Returned is the 'filepath'. Enter this as the argument of 'load' to later reload the model. Explicitly, that is
    'FoKLRoutines.load(filepath)' or 'FoKLRoutines.load(filename, directory)'.
    Note the directory must exist prior to calling this method.
    """
    if filename is None:
        t = time.gmtime()
        def two_digits(a):
            if a < 10:
                a = "0" + str(a)
            else:
                a = str(a)
            return a
        ymd = [str(t[0])]
        for i in range(1, 6):
            ymd.append(two_digits(t[i]))
        t_str = ymd[0] + ymd[1] + ymd[2] + ymd[3] + ymd[4] + ymd[5]
        filename = "model_" + t_str + ".fokl"
    elif filename[-5::] != ".fokl":
        filename = filename + ".fokl"
    if directory is not None:
        filepath = os.path.join(directory, filename)
    else:
        filepath = filename
    file = open(filepath, "wb")
    pickle.dump(self, file)
    file.close()
    time.sleep(1)  # so that next saved model is guaranteed a different filename
    return filepath
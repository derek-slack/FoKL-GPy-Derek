def clear(self, keep=None, clear=None, all=False):
    """
    Delete all attributes from the FoKL class except for hyperparameters and settings by default, but user may
    specify otherwise. If an attribute is listed in both 'clear' and 'keep', then the attribute is cleared.

    Optional Inputs:
        keep (list of strings)  == additional attributes to keep, e.g., ['mtx']
        clear (list of strings) == hyperparameters to delete, e.g., ['kernel', 'phis']
        all (boolean)           == if True then all attributes (including hyperparameters) get deleted regardless

    Tip: To remove all attributes, simply call 'self.clear(all=1)'.
    """

    if all is not False:  # if not default
        all = _str_to_bool(all)  # convert to boolean if all='on', etc.

    if all is False:
        attrs_to_keep = self.keep  # default
        if isinstance(keep, list) or isinstance(keep, str):  # str in case single entry (e.g., keep='mtx')
            attrs_to_keep += keep  # add user-specified attributes to list of ones to keep
            attrs_to_keep = list(np.unique(attrs_to_keep))  # remove duplicates
        if isinstance(clear, list) or isinstance(clear, str):
            for attr in clear:
                attrs_to_keep.remove(attr)  # delete attribute from list of ones to keep
    else:  # if all=True
        attrs_to_keep = []  # empty list so that all attributes get deleted

    attrs = list(vars(self).keys())  # list of all currently defined attributes
    for attr in attrs:
        if attr not in attrs_to_keep:
            delattr(self, attr)  # delete attribute from FoKL class if not keeping

    return
def evaluate_basis(self, c, x, kernel=None, d=0):
    """
    Evaluate a basis function at a single point by providing coefficients, x value(s), and (optionally) the kernel.
    Inputs:
        > c == coefficients of a single basis functions
        > x == value of independent variable at which to evaluate the basis function
    Optional Input:
        > kernel == 'Cubic Splines' or 'Bernoulli Polynomials' == self.kernel (default)
        > d      == integer representing order of derivative   == 0 (default)
    Output (in Python syntax, for d=0):
        > if kernel == 'Cubic Splines':
            > basis = c[0] + c[1]*x + c[2]*(x**2) + c[3]*(x**3)
        > if kernel == 'Bernoulli Polynomials':
            > basis = sum(c[k]*(x**k) for k in range(len(c)))
    """
    if kernel is None:
        kernel = self.kernel
    elif isinstance(kernel, int):
        kernel = self.kernels[kernel]
    if kernel not in self.kernels:  # check user's provided kernel is supported
        raise ValueError(f"The kernel {kernel} is not currently supported. Please select from the following: "
                         f"{self.kernels}.")
    if kernel == self.kernels[0]:  # == 'Cubic Splines':
        if d == 0:  # basis function
            basis = c[0] + c[1] * x + c[2] * (x ** 2) + c[3] * (x ** 3)
        elif d == 1:  # first derivative
            basis = c[1] + 2 * c[2] * x + 3 * c[3] * (x ** 2)
        elif d == 2:  # second derivative
            basis = 2 * c[2] + 6 * c[3] * x
    elif kernel == self.kernels[1]:  # == 'Bernoulli Polynomials':
        if d == 0:  # basis function
            basis = c[0] + sum(c[k] * (x ** k) for k in range(1, len(c)))
        elif d == 1:  # first derivative
            basis = c[1] + sum(k * c[k] * (x ** (k - 1)) for k in range(2, len(c)))
        elif d == 2:  # second derivative
            basis = sum((k - 1) * k * c[k] * (x ** (k - 2)) for k in range(2, len(c)))
    return basis
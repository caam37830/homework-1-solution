"""
A library of functions
"""
import numpy as np
import matplotlib.pyplot as plt
import numbers

class AbstractFunction:
    """
    An abstract function class
    """

    def derivative(self):
        """
        returns another function f' which is the derivative of x
        """
        raise NotImplementedError("derivative")


    def __str__(self):
        return "AbstractFunction"


    def __repr__(self):
        return "AbstractFunction"


    def evaluate(self, x):
        """
        evaluate at x

        assumes x is a numeric value, or numpy array of values
        """
        raise NotImplementedError("evaluate")


    def __call__(self, x):
        """
        if x is another AbstractFunction, return the composition of functions

        if x is a string return a string that uses x as the indeterminate

        otherwise, evaluate function at a point x using evaluate
        """
        if isinstance(x, AbstractFunction):
            return Compose(self, x)
        elif isinstance(x, str):
            return self.__str__().format(x)
        else:
            return self.evaluate(x)


    # the rest of these methods will be implemented when we write the appropriate functions
    def __add__(self, other):
        """
        returns a new function expressing the sum of two functions
        """
        return Sum(self, other)


    def __mul__(self, other):
        """
        returns a new function expressing the product of two functions
        """
        return Product(self, other)


    def __neg__(self):
        return Scale(-1)(self)


    def __truediv__(self, other):
        return self * other**-1


    def __pow__(self, n):
        return Power(n)(self)


    def plot(self, vals=np.linspace(-1,1,100), **kwargs):
        """
        plots function on values
        pass kwargs to plotting function
        """
        y = self.evaluate(vals)
        fig = plt.plot(vals, y, **kwargs)
        return fig


    def taylor_series(self, x0, deg=5):
        """
        Returns the Taylor series of f centered at x0 truncated to degree k.
        """
        Tf = Constant(self(x0))
        var = Affine(1, -x0) # x - x0
        fk = self.derivative()
        fact = 1 # holds factorial(k)
        for k in range(1,deg+1):
            fact = fact * k
            Tf = Tf + Constant(fk(x0) / fact) * var**k
            fk = fk.derivative() # increase derivative

        return Tf




class Polynomial(AbstractFunction):
    """
    polynomial c_n x^n + ... + c_1 x + c_0
    """

    def __init__(self, *args):
        """
        Polynomial(c_n ... c_0)

        Creates a polynomial
        c_n x^n + c_{n-1} x^{n-1} + ... + c_0
        """
        self.coeff = np.array(list(args))


    def __repr__(self):
        return "Polynomial{}".format(tuple(self.coeff))


    def __str__(self):
        """
        We'll create a string starting with leading term first

        there are a lot of branch conditions to make everything look pretty
        """
        s = ""
        deg = self.degree()
        for i, c in enumerate(self.coeff):
            if i < deg-1:
                if c == 0:
                    # don't print term at all
                    continue
                elif c == 1:
                    # supress coefficient
                    s = s + "({{0}})^{} + ".format(deg - i)
                else:
                    # print coefficient
                    s = s + "{}({{0}})^{} + ".format(c, deg - i)
            elif i == deg-1:
                # linear term
                if c == 0:
                    continue
                elif c == 1:
                    # suppress coefficient
                    s = s + "{0} + "
                else:
                    s = s + "{}({{0}}) + ".format(c)
            else:
                if c == 0 and len(s) > 0:
                    continue
                else:
                    # constant term
                    s = s + "{}".format(c)

        # handle possible trailing +
        if s[-3:] == " + ":
            s = s[:-3]

        return s


    def evaluate(self, x):
        """
        evaluate polynomial at x
        """
        if isinstance(x, numbers.Number):
            ret = 0
            for k, c in enumerate(reversed(self.coeff)):
                ret = ret + c * x**k
            return ret
        elif isinstance(x, np.ndarray):
            x = np.array(x)
            # use vandermonde matrix
            return np.vander(x, len(self.coeff)).dot(self.coeff)


    def derivative(self):
        if len(self.coeff) == 1:
            return Polynomial(0)
        return Polynomial(*(self.coeff[:-1] * np.array([n+1 for n in reversed(range(self.degree()))])))


    def degree(self):
        return len(self.coeff) - 1


    def __add__(self, other):
        """
        Polynomials are closed under addition - implement special rule
        """
        if isinstance(other, Polynomial):
            # add
            if self.degree() > other.degree():
                coeff = self.coeff
                coeff[-(other.degree() + 1):] += other.coeff
                return Polynomial(*coeff)
            else:
                coeff = other.coeff
                coeff[-(self.degree() + 1):] += self.coeff
                return Polynomial(*coeff)

        else:
            # do default add
            return super().__add__(other)


    def __mul__(self, other):
        """
        Polynomials are clused under multiplication - implement special rule
        """
        if isinstance(other, Polynomial):
            return Polynomial(*np.polymul(self.coeff, other.coeff))
        else:
            return super().__mul__(other)


class Affine(Polynomial):
    """
    affine function a * x + b
    """
    def __init__(self, a, b):
        super().__init__(a, b)


## Implement Scale and Polynomial
class Scale(Polynomial):
    def __init__(self, a):
        super().__init__(a, 0)


class Constant(Polynomial):
    def __init__(self, c):
        super().__init__(c)


## Implement Compose, Product, and Sum
class Sum(AbstractFunction):
    """
    f + g
    """
    def __init__(self, f, g):
        if isinstance(f, AbstractFunction) and isinstance(g, AbstractFunction):
            self.f = f
            self.g = g
        else:
            raise AssertionError("must input AbstractFunction functions")

    def __str__(self):
        return "{} + {}".format(self.f, self.g)

    def __repr__(self):
        return "Sum({}, {})".format(self.f.__repr__(), self.g.__repr__())

    def evaluate(self, x):
        return self.f(x) + self.g(x)

    def derivative(self):
        return Sum(self.f.derivative(), self.g.derivative())


class Product(AbstractFunction):
    """
    f * g
    """
    def __init__(self, f, g):
        if isinstance(f, AbstractFunction) and isinstance(g, AbstractFunction):
            self.f = f
            self.g = g
        else:
            raise AssertionError("must input AbstractFunction functions")

    def __str__(self):
        return "({}) * ({})".format(self.f, self.g)

    def __repr__(self):
        return "Product({}, {})".format(self.f.__repr__(), self.g.__repr__())

    def evaluate(self, x):
        return self.f(x) * self.g(x)

    def derivative(self):
        # product rule
        return self.f.derivative() * self.g + self.f * self.g.derivative()


class Compose(AbstractFunction):
    """
    composition of functions f \\circ g
    """

    def __init__(self, f, g):
        if isinstance(f, AbstractFunction) and isinstance(g, AbstractFunction):
            self.f = f
            self.g = g
        else:
            raise AssertionError("must input AbstractFunction functions")

    def __str__(self):
        return "{}".format(self.f).format(self.g)

    def __repr__(self):
        return "Compose({}, {})".format(self.f.__repr__(), self.g.__repr__())

    def evaluate(self, x):
        return self.f(self.g(x))

    def derivative(self):
        # chain rule
        return self.g.derivative() * self.f.derivative()(self.g)


## Implement Power, Log, Exponential, Sin, Cos
class Power(AbstractFunction):
    """
    function x^n

    n can be negative or non-integer
    """

    def __init__(self, n):
        self.n = n

    def __str__(self):
        return "({{0}})^{}".format(self.n)

    def __repr__(self):
        return "Power({})".format(self.n)

    def derivative(self):
        return Scale(self.n)(Power(self.n-1))

    def evaluate(self, x):
        return np.power(x, self.n)



class Exponential(AbstractFunction):
    """
    function exp(x)
    """

    def __init__(self):
        # nothing to do
        return

    def __str__(self):
        return "exp({0})"

    def __repr__(self):
        return "Exponential()"

    def derivative(self):
        return Exponential()

    def evaluate(self, x):
        return np.exp(x)


class Sin(AbstractFunction):
    """
    function sin(x)
    """

    def __init__(self):
        return

    def __str__(self):
        return "sin({0})"

    def __repr__(self):
        return "Sin()"

    def derivative(self):
        return Cos()

    def evaluate(self, x):
        return np.sin(x)


class Cos(AbstractFunction):
    """
    function cos(x)
    """

    def __init__(self):
        return

    def __str__(self):
        return "cos({0})"

    def __repr__(self):
        return "Cos()"

    def derivative(self):
        return Constant(-1) * Sin()

    def evaluate(self, x):
        return np.cos(x)


class Log(AbstractFunction):
    """
    function log(x)
    """
    def __init__(self):
        return

    def __str__(self):
        return "log({0})"

    def __repr__(self):
        return "Log()"

    def derivative(self):
        return Power(-1)

    def evaluate(self, x):
        return np.log(x)


class Symbolic(AbstractFunction):
    def __init__(self, name):
        if isinstance(name, str):
            self.name=name
        else:
            raise AssertionError("name must be string")

    def __str__(self):
        return self.name + "({0})"

    def __repr__(self):
        return "Symbolic({})".format(self.name)

    def evaluate(self, x):
        return self.name + "({0})".format(x)

    # product rule (f*g)' = f'*g + f*g'
    def derivative(self):
        return Symbolic(self.name + "'")


def newton_root(f, x0, tol=1e-8):
    """
    find a point x so that f(x) is close to 0,
    measured by abs(f(x)) < tol

    Use Newton's method starting at point x0
    """
    if not isinstance(f, AbstractFunction):
        raise AssertionError("f must be an AbstractFunction")
    if isinstance(f, Symbolic):
        raise AssertionError("f can not be Symbolic")

    f1 = f.derivative()
    x = x0
    fx = f(x)
    while abs(fx) > tol:
        x = x - fx / f1(x)
        fx = f(x)

    return x


def newton_extremum(f, x0, tol=1e-8):
    """
    find a point x which is close to a local maximum or minimum of f,
    measured by abs(f'(x)) < tol

    Use Newton's method starting at point x0
    """
    return newton_root(f.derivative(), x0)

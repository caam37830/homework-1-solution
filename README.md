# Homework 1 - Object Oriented Programming and Functions

In this homework, you'll create a library of function classes which are related by class inheritance.

You can run
```
conda install --file requirements.txt
```
to install the packages in [`requirements.txt`](requirements.txt)

Please put plots and written answers in the Jupyter notebook [`answers.ipynb`](answers.ipynb)

## Important Information

### Due Date
This assignment is due Friday, October 16 at 12pm (noon) Chicago time.


### Grading Rubric

The following rubric will be used for grading.

|   | Autograder | Correctness | Style | Total |
|:-:|:-:|:-:|:-:|:-:|
| Survey    |   | /10 | /0 |  /10 |
| Problem 0 |   | /8  | /2  |  /10 |
| Problem 1 | /5 | /10  | /5  | /20 |
| Problem 2 | /10 | /12  | /3  | /25 |
| Problem 3 | /5 | /12  | /3  | /20 |
| Problem 4 |   |  /10 | /5  | /15 |

Correctness will be based on code (i.e. did you provide was was asked for) and the content of [`answers.md`](answers.md).

To get full points on style you should use comments to explain what you are doing in your code and write docstrings for your functions.  In other words, make your code readable and understandable.

### Autograder

You can run tests for the functions you will write in problems 1-3 using either `unittest` or `pytest` (you may need to `conda install pytest`).  From the command line:
```
python -m unittest test.py
```
or
```
pytest test.py
```
The tests are in [`test.py`](test.py).  You do not need to modify (or understand) this code.

You need to pass all tests to receive full points.

## Problem 0

In this problem, you will implement a library of function classes related by class inheritance.

We want this library of functions to behave nicely with NumPy arrays.  The `evaluate` method should be compatible with NumPy element-wise operations.

Put all your class and function definitions in [`functions.py`](functions.py), and use [`answers.ipynb`](answers.ipynb) for written answers, plots, or displaying output of expressions.

### Part A: Finish the parent class (5 points)

The parent class for this library is `AbstractFunction`.  Implement
Complete the `plot` method in this class.  You should return the output of the `plot` function in PyPlot.

Use the `evaluate` method on the keyword argument `vals` to make a plot of the function (evaluate will be implemented in child classes).  Pass any additional keyword arguments to the `plot` function in PyPlot.

Note that we will need to implement child classes:
* `Compose` (composition of functions)
* `Sum` (sum of functions)
* `Product` (product of functions)
* `Scale` (scaling function)
* `Power` (function that applies a power)

Some additional notes on methods which will be defined in child classes:
* `evaluate` - evaluate the function on a value, or element-wise on a numpy array
* `derivative` - return another child class of `AbstractFunction` as the derivative of the function.

The `__call__` method allows you to use an object as a function in Python.  For example, if `f` is some derived class of `AbstractFunction`, `f(x)` will use the `__call__` method.  You can see that this method does different things depending on what the type of `x` is.

Test your function by plotting `5*x^2 + 3*x + 1` using the provided Polynomial class
```python
p = Polynomial(5,3,1)
p.plot(color='red')
```

### Part B: Implement Scale and Constant functions (5 points)

A class for Polynomial functions, named `Polynomial` has been provided in `functions.py`.  This is more complex than anything you'll need to do.

When implementing derived classes, you can access methods of the parent class (even as you specialize them) using `super()`.  See the definition of `Affine` for an example of how we can implement the `__init__` method for functions of the form `a*x + b` using the `__init__` method of `Polynomial`.

Implement classes `Scale` and `Constant` as child classes of `Polynomial`.  You should only need to implement the`__init__` method for both classes.

`Scale(a)` should be equivalent to the polynomial `a * x + 0`

`Constant(c)` should be equivalent to the polynomial `c`

Make plots of `Scale(2)` and `Constant(1)` using the `plot` method.

### Part C: Implement Compose, Product, and Sum functions (20 points)

Implement child classes of `AbstractFunction`:
1. `Compose`, where `Compose(f, g)(x)` acts as `f(g(x))`
2. `Product`, where `Product(f, g)(x)` acts as `f(x) * g(x)`
3. `Sum`, where `Sum(f,g)(x)` acts as `f(x) + g(x)`

You should provide implementations for:
* `__init__`
* `__str__`
* `__repr__`
* `derivative` (recall chain rule and product rule)
* `evaluate`

When implementing `__str__`, place `{0}` where indeterminates in the function would go.
You can look at the implementation of `Polynomial` for examples of this.  If you call the `format` method on the string, you need to escape the sequence (so it isn't formatted), by enclosing in an extra set of braces: `"{{0}}"`

### Part D: Implement Power (and some other functions) (20 points)

Implement additional classes of `AbstractFunction`
1. `Power`: `Power(n)(x)` acts as `x**n` (n can be negative, or non-integer)
2. `Log`: `Log()(x)` acts as `np.log(x)`
3. `Exponential`: `Exponential()(x)` acts as `np.exp(x)`
4. `Sin`: `Sin()(x)` acts as `np.sin(x)`
5. `Cos`: `Cos()(x)` acts as `np.cos(x)`

You should provide implementations for:
* `__init__`
* `__str__`
* `__repr__`
* `derivative`
* `evaluate`

When implementing `__str__`, place `{0}` where indeterminates in the function would go.
You can look at the implementation of `Polynomial` for examples of this.

Make plots of `Sin()`, `Cos()`, and `Exponential()`.


### Part E: Symbolic Functions

Implement a class for symbolic functions.  The data for a symbolic function is a string, which is the name of the function.  For example, we should be able to define a symbolic function
```python
f = Symbolic('f')
```
The string method should output:
```python
str(f)
"f({0})"
```
The evaluate method should just print a string with whatever the input is, so when the `__call__` method is used, we have
```python
f(5)
"f(5)"
```
The derivative method should add an apostrophe to the end of the name:
```python
f.derivative()
Symbolic("f'")
```

Note that Symbolic functions won't be compatible with the `plot` method defined in the `AbstractFunction` class.

### Part F: Use the Module


## Problem 1

### Newton's method for root finding
Implement Newton's method for root finding using the call signature
```python
def newton_root(f, x0, tol=1e-8):
    """
    find a point x so that f(x) is close to 0,
    measured by abs(f(x)) < tol

    Use Newton's method starting at point x0
    """
```
The function should assume that `f` is an `AbstractFunction`, and that `x` is a real number.  Put in a type check to verify that `f` is an `AbstractFunction` but it is not `Symbolic`.

Implement this function in `functions.py`

Find a root of `sin(exp(x))` starting at `x0=1.0`. Make a plot of `sin(exp(x))` and visualize the root using `plt.scatter`.

### Newton's method for finding extrema

Implement a function that finds a local extremum for a function using the call signature
```python
def newton_extremum(f, x0, tol=1e-8):
    """
    find a point x which is close to a local maximum or minimum of f,
    measured by abs(f'(x)) < tol

    Use Newton's method starting at point x0
    """
```
Again, assume that `f` is an `AbstractFunction`, and `x` is a real number.

Implement this function in `functions.py`

Find a minimum or maximum of `sin(exp(x))` starting at `x0=0.0`. Make a plot of `sin(exp(x))` and visualize the extremum using `plt.scatter`.

## Problem 2 - Taylor Series

### Part A - add a method to AbstractFunction
Recall the Taylor series of a function `f` centered at `x0` is the polynomial
```
Tf = f(x0) + f'(x0)(x - x0) + 1/2 * f''(x0)(x - x0)**2 + ...
```
Add a method `taylor_series` to the `AbstractFunction` base class
```python
def taylor_series(self, x0, deg=5):
    """
    Returns the Taylor series of f centered at x0 truncated to degree k.
    """
```
The return type should be another `AbstractFunction`.

### Part B - use your Taylor series

Make a plot that displays `sin(x)` as well as its degree-k taylor series for `k in [0,1,3,5]` on the interval `[-3,3]`

Your plot should include labels for each line displayed, as well as a legend.

## Feedback

If you'd like share how long it took you to complete this assignment, it will help adjust the difficulty for future assignments.  You're welcome to share additional feedback as well.

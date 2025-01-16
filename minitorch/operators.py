"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.


# Mathematical functions:
# - mul
def mul(a: float, b: float) -> float:
    """Calculate a times b"""
    return a * b


# - id
def id(a: float) -> float:
    """Implement the identity function"""
    return a


# - add
def add(a: float, b: float) -> float:
    """Calculate a plus b"""
    return a + b


# - add
def sub(a: float, b: float) -> float:
    """Calculate a minus b"""
    return a - b


# - neg
def neg(a: float) -> float:
    """Return -1 times a"""
    return -a


# - gt
def gt(a: float, b: float) -> float:
    """Return True if a is less than b; return False otherwise"""
    return 1.0 if a > b else 0.0


# - lt
def lt(a: float, b: float) -> float:
    """Return True if a is less than b; return False otherwise"""
    return 1.0 if a < b else 0.0


# - eq
def eq(a: float, b: float) -> float:
    """Return True if a is equal to b; return False otherwise"""
    return 1.0 if a == b else 0.0


# - max
def max(a: float, b: float) -> float:
    """Calculate the max of two values a and b. Return b if it is greater than
    a; otherwise, return a.
    """
    return a if a > b else b


# For is_close:
# $f(x) = |x - y| < 1e-2$
# - is_close
def is_close(a: float, b: float) -> bool:
    """Return True if the absolute difference between a and b is <1E-2; return
    False otherwise.
    """
    return ((a - b) < 1e-2) and ((b - a) < 1e-2)


#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# - sigmoid
def sigmoid(x: float) -> float:
    """Implement the sigmoid function"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def sigmoid_back(x: float, d_out: float) -> float:
    """Implement the sigmoid function"""
    if x >= 0:
        return d_out * math.exp(-x) / math.pow(1.0 + math.exp(-x), 2)
    else:
        return d_out * math.exp(x) / math.pow(1.0 + math.exp(x), 2)


# - relu
def relu(x: float) -> float:
    """Implement the relu function"""
    return x if x > 0 else 0.0


EPS = 1e-6


# - log
def log(x: float) -> float:
    """Implement the natural logarithm function, reusing Python's built-in
    natural logarithm function.
    """
    return math.log(x + EPS)


# - exp
def exp(x: float) -> float:
    """Calculate e to the power of x, i.e. the natural exponential function."""
    return math.exp(x)


# - log_back
def log_back(a: float, b: float) -> float:
    """Return b times the derivative with respect to a of log(a)"""
    return b / (a + EPS)


# - inv
def inv(x: float) -> float:
    """Implement the inverse function."""
    return 1.0 / x


# - inv_back
def inv_back(a: float, b: float) -> float:
    """Return b times the derivative with respect to a of 1/a"""
    return -(1.0 / a**2) * b


# - relu_back
def relu_back(a: float, b: float) -> float:
    """Return b times the derivative with respect to a of relu(a)"""
    if a < 0:
        return 0.0
    return b


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Higher-order map.

    See https://en.wikipedia.org/wiki/Map_(higher-order-function)

    Args:
    ----
        fn: Function from one value to one value

    Returns:
    -------
        A function that takes a list, applies `fn` to each element, and returns a
        new list

    """

    def _map(ls: Iterable[float]) -> Iterable[float]:
        ret = []
        for x in ls:
            ret.append(fn(x))
        return ret

    return _map


def negList(a: Iterable[float]) -> Iterable[float]:
    """Return a new iterable containing -1 * a[i] at each index i in a"""
    return map(neg)(a)


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    r"""Higher-order reduce.

    Args:
    ----
        fn: combine two values
        start: start value $x_0$

    Returns:
    -------
        Function that takes a list `ls` of elements
        $x_1 \ldots x_n$ and computes the reduction :math:`fn(x_3, fn(x_2,
        fn(x_1, x_0)))`

    """

    def _reduce(ls: Iterable[float]) -> float:
        val = start
        for l in ls:
            val = fn(val, l)
        return val

    return _reduce


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Higher-order zipwith (or map2).

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
    ----
        fn: combine two values

    Returns:
    -------
        Function that takes two equally sized lists `ls1` and `ls2`, and
        produces a new list applying fn(x, y) on each pair of elements.

    """

    def _zipWith(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        ret = []
        for x, y in zip(ls1, ls2):
            ret.append(fn(x, y))
        return ret

    return _zipWith


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Return a new iterable whose ith value is a_i + b_i for the ith values in a and b"""
    return zipWith(add)(ls1, ls2)


def sum(ls: Iterable[float]) -> float:
    """Return the sum all elements of the list a, where 0.0 is the default value"""
    return reduce(add, 0.0)(ls)


def prod(ls: Iterable[float]) -> float:
    """Return the product all elements of the list a, where 1.0 is the default value"""
    return reduce(mul, 1.0)(ls)

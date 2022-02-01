"""
Collection of the core mathematical operators used throughout the code base.
"""


import math

# ## Task 0.1

# Implementation of a prelude of elementary functions.


def mul(x, y):
    """
    Multiplication of floats, f(x,y) = x * y

    Args:
        x (float): A floating point number
        y (float): A floating point number

    Returns:
        float: Value of x * y
    """

    return x * y

def id(x):
    """
    Identity function, f(x) = x

    Args:
        x (float): A floating point number

    Returns:
        float: The input number x
    """

    return x


def add(x, y):
    """
    Addition of floats, f(x, y) = x + y

    Args:
        x (float): A floating point number
        y (float): A floating point number
    
    Returns:
        float: The value of x + y
    """
    return x + y


def neg(x):
    """
    Negation of floats, f(x) = -x

    Args:
        x (float): A floating point number
    
    Returns:
        float: The negation of x
    """
    return -x


def lt(x, y):
    """
    Less than comparison, f(x,y) = x < y

    Args:
        x (float): A floating point number
        y (float): A floating point number

    Returns:
        float: 1.0 if x lt y, else 0.0
    """
    if x > y:
        return 0.0
    elif x < y:
        return 1.0


def eq(x, y):
    """
    Equality comparison, f(x,y) = (x == y)

    Args:
        x (float): A floating point number
        y (float): A floating point number
    Returns:
        float: 1.0 if x equals y, else 0.0
    """
    
    if x == y:
        return 1.0
    else:
        return 0.0


def max(x, y):
    """
    Returns the largest of two floats, f(x,y) = max(x,y)

    Args:
        x (float): A floating point number
        y (float): A floating point number

    Returns:
        float: x or y with largest value
    """
    if x > y:
        return x
    else:
        return y


def is_close(x, y):
    """
    Is close comparison for floats

    Args:
        x (float): A floating point number
        y (float): A floating point number

    Returns:
        float: 1.0 if x is close to y, else 0.0
    """
    if abs(x - y) > 1e-2:
        return 0.0
    else:
        return 1.0


def sigmoid(x):
    r"""
    :math:`f(x) =  \frac{1.0}{(1.0 + e^{-x})}`

    (See `<https://en.wikipedia.org/wiki/Sigmoid_function>`_ .)

    Calculate as

    :math:`f(x) =  \frac{1.0}{(1.0 + e^{-x})}` if x >=0 else :math:`\frac{e^x}{(1.0 + e^{x})}`

    for stability.

    Args:
        x (float): input

    Returns:
        float : sigmoid value
    """
    if x >= 0.0:
        return 1.0 / (1.0 + exp(-x))
    else:
        return exp(x) / (1.0 + exp(x))


def relu(x):
    """
    :math:`f(x) =` x if x is greater than 0, else 0

    (See `<https://en.wikipedia.org/wiki/Rectifier_(neural_networks)>`_ .)

    Args:
        x (float): input

    Returns:
        float : relu value
    """
    if x > 0.0:
        return x
    else:
        return 0.0


EPS = 1e-6


def log(x):
    ":math:`f(x) = log(x)`"
    return math.log(x + EPS)


def exp(x):
    ":math:`f(x) = e^{x}`"
    return math.exp(x)


def log_back(x, d):
    """ d * f'(x) when f is log(x)

    Args:
        x (float): 
        d (float):

    Return:
        float:


    """
    return d * (1.0 / log(x))


def inv(x):
    """Inverse function, f(x) = 1 / x

    Args:
        x (float): A floating point number

    Returns:
        float: The inverse of x
    """
    return 1.0 / x


def inv_back(x, d):
    """ d * f'(x) when f is 1 / x

    Args:
        x (float): 
        d (float):
    Return:
        float:

    """
    return d * ( -1.0 /( log(x) ** 2))


def relu_back(x, d):
    """ d * f'(x) when f is relu

    Args:
        x (float): 
        d (float):
    Return:
        float:

    """
    if x > 0.0:
        return d * 1.0
    else:
        return 0.0

# ## Task 0.3

# Small library of elementary higher-order functions for practice.


def map(fn):
    """
    Higher-order map.

    .. image:: figs/Ops/maplist.png


    See `<https://en.wikipedia.org/wiki/Map_(higher-order_function)>`_

    Args:
        fn (one-arg function): Function from one value to one value.

    Returns:
        function : A function that takes a list, applies `fn` to each element, and returns a
        new list
    """
    def function(list):
        output = []
        for element in list:
            res = fn(element)
            output.append(res)
        return output

    return function
    


def negList(ls):
    "Use :func:`map` and :func:`neg` to negate each element in `ls`"
    function = map(neg)
    return function(ls)


def zipWith(fn):
    """
    Higher-order zipwith (or map2).

    .. image:: figs/Ops/ziplist.png

    See `<https://en.wikipedia.org/wiki/Map_(higher-order_function)>`_

    Args:
        fn (two-arg function): combine two values

    Returns:
        function : takes two equally sized lists `ls1` and `ls2`, produce a new list by
        applying fn(x, y) on each pair of elements.

    """

    def function(list1, list2):
        output = []
        for item1, item2 in zip(list1, list2):
            res = fn(item1, item2)
            output.append(res)
        return output

    return function



def addLists(ls1, ls2):
    "Add the elements of `ls1` and `ls2` using :func:`zipWith` and :func:`add`"
    function = zipWith(add)

    return function(ls1, ls2)
    


def reduce(fn, start):
    r"""
    Higher-order reduce.

    .. image:: figs/Ops/reducelist.png


    Args:
        fn (two-arg function): combine two values
        start (float): start value :math:`x_0`

    Returns:
        function : function that takes a list `ls` of elements
        :math:`x_1 \ldots x_n` and computes the reduction :math:`fn(x_3, fn(x_2,
        fn(x_1, x_0)))`
    """
    def function(list):
        cum_val = start
        for value in list:
            cum_val = fn(cum_val, value)
        return cum_val

    return function


def sum(ls):
    "Sum up a list using :func:`reduce` and :func:`add`."

    function = reduce(add, 0)

    return function(ls)


def prod(ls):
    "Product of a list using :func:`reduce` and :func:`mul`."
    
    function = reduce(mul, 1)

    return function(ls)

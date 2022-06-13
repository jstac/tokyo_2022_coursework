---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# NumPy 


## Overview

[NumPy](https://en.wikipedia.org/wiki/NumPy) is a first-rate library for numerical programming

* Widely used in academia, finance and industry.
* Mature, fast, stable and under continuous development.

We have already seen some code involving NumPy in the preceding lectures.

In this lecture, we will start a more systematic discussion of both

* NumPy arrays and
* the fundamental array processing operations provided by NumPy.

+++

### References

* [The official NumPy documentation](http://docs.scipy.org/doc/numpy/reference/).

+++

## NumPy Arrays

The essential problem that NumPy solves is fast array processing.

The most important structure that NumPy defines is an array data type formally called a [numpy.ndarray](http://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html).

NumPy arrays power a large proportion of the scientific Python ecosystem.

Let's first import the library.

```{code-cell} python3
import numpy as np
```

To create a NumPy array containing only zeros we use  [np.zeros](http://docs.scipy.org/doc/numpy/reference/generated/numpy.zeros.html#numpy.zeros)

```{code-cell} python3
a = np.zeros(3)
a
```

```{code-cell} python3
type(a)
```

NumPy arrays are somewhat like native Python lists, except that

* Data *must be homogeneous* (all elements of the same type).
* These types must be one of the [data types](https://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html) (`dtypes`) provided by NumPy.

The most important of these dtypes are:

* float64: 64 bit floating-point number
* int64: 64 bit integer
* bool:  8 bit True or False

There are also dtypes to represent complex numbers, unsigned integers, etc.

On modern machines, the default dtype for arrays is `float64`

```{code-cell} python3
a = np.zeros(3)
type(a[0])
```

If we want to use integers we can specify as follows:

```{code-cell} python3
a = np.zeros(3, dtype=int)
type(a[0])
```

### Shape and Dimension

Consider the following assignment

```{code-cell} python3
z = np.zeros(10)
```

Here `z` is a *flat* array with no dimension --- neither row nor column vector.

The dimension is recorded in the `shape` attribute, which is a tuple

```{code-cell} python3
z.shape
```

Here the shape tuple has only one element, which is the length of the array (tuples with one element end with a comma).

To give it dimension, we can change the `shape` attribute

```{code-cell} python3
z.shape = (10, 1)
z
```

```{code-cell} python3
z = np.zeros(4)
z.shape = (2, 2)
z
```

In the last case, to make the 2 by 2 array, we could also pass a tuple to the `zeros()` function, as
in `z = np.zeros((2, 2))`.

### Creating Arrays

As we've seen, the `np.zeros` function creates an array of zeros.

You can probably guess what `np.ones` creates.

Related is `np.empty`, which creates arrays in memory that can later be populated with data

```{code-cell} python3
z = np.empty(3)
z
```

The numbers you see here are garbage values.

(Python allocates 3 contiguous 64 bit pieces of memory, and the existing contents of those memory slots are interpreted as `float64` values)

To set up a grid of evenly spaced numbers use `np.linspace`

```{code-cell} python3
z = np.linspace(2, 4, 5)  # From 2 to 4, with 5 elements
```

To create an identity matrix use either `np.identity` or `np.eye`

```{code-cell} python3
z = np.identity(2)
z
```

In addition, NumPy arrays can be created from Python lists, tuples, etc. using `np.array`

```{code-cell} python3
z = np.array([10, 20])                 # ndarray from Python list
z
```

```{code-cell} python3
type(z)
```

```{code-cell} python3
z = np.array((10, 20), dtype=float)    # Here 'float' is equivalent to 'np.float64'
z
```

```{code-cell} python3
z = np.array([[1, 2], [3, 4]])         # 2D array from a list of lists
z
```

See also `np.asarray`, which performs a similar function, but does not make
a distinct copy of data already in a NumPy array.

```{code-cell} python3
na = np.linspace(10, 20, 2)
na is np.asarray(na)   # Does not copy NumPy arrays
```

```{code-cell} python3
na is np.array(na)     # Does make a new copy --- perhaps unnecessarily
```

To read in the array data from a text file containing numeric data use `np.loadtxt`
or `np.genfromtxt`---see [the documentation](http://docs.scipy.org/doc/numpy/reference/routines.io.html) for details.

### Array Indexing

For a flat array, indexing is the same as Python sequences:

```{code-cell} python3
z = np.linspace(1, 2, 5)
z
```

```{code-cell} python3
z[0]
```

```{code-cell} python3
z[0:2]  # Two elements, starting at element 0
```

```{code-cell} python3
z[-1]
```

For 2D arrays the index syntax is as follows:

```{code-cell} python3
z = np.array([[1, 2], [3, 4]])
z
```

```{code-cell} python3
z[0, 0]
```

```{code-cell} python3
z[0, 1]
```

And so on.

Note that indices are still zero-based, to maintain compatibility with Python sequences.

Columns and rows can be extracted as follows

```{code-cell} python3
z[0, :]
```

```{code-cell} python3
z[:, 1]
```

NumPy arrays of integers can also be used to extract elements

```{code-cell} python3
z = np.linspace(2, 4, 5)
z
```

```{code-cell} python3
indices = np.array((0, 2, 3))
z[indices]
```

Finally, an array of `dtype bool` can be used to extract elements

```{code-cell} python3
z
```

```{code-cell} python3
d = np.array([0, 1, 1, 0, 0], dtype=bool)
d
```

```{code-cell} python3
z[d]
```

We'll see why this is useful below.

An aside: all elements of an array can be set equal to one number using slice notation

```{code-cell} python3
z = np.empty(3)
z
```

```{code-cell} python3
z[:] = 42
z
```

### Array Methods

Arrays have useful methods, all of which are carefully optimized

```{code-cell} python3
a = np.array((4, 3, 2, 1))
a
```

```{code-cell} python3
a.sort()              # Sorts a in place
a
```

```{code-cell} python3
a.sum()               # Sum
```

```{code-cell} python3
a.mean()              # Mean
```

```{code-cell} python3
a.max()               # Max
```

```{code-cell} python3
a.argmax()            # Returns the index of the maximal element
```

```{code-cell} python3
a.cumsum()            # Cumulative sum of the elements of a
```

```{code-cell} python3
a.cumprod()           # Cumulative product of the elements of a
```

```{code-cell} python3
a.var()               # Variance
```

```{code-cell} python3
a.std()               # Standard deviation
```

```{code-cell} python3
a.shape = (2, 2)
a.T                   # Equivalent to a.transpose()
```

Another method worth knowing is `searchsorted()`.

If `z` is a nondecreasing array, then `z.searchsorted(a)` returns the index of the first element of `z` that is `>= a`

```{code-cell} python3
z = np.linspace(2, 4, 5)
z
```

```{code-cell} python3
z.searchsorted(2.2)
```

Many of the methods discussed above have equivalent functions in the NumPy namespace

```{code-cell} python3
a = np.array((4, 3, 2, 1))
```

```{code-cell} python3
np.sum(a)
```

```{code-cell} python3
np.mean(a)
```

## Operations on Arrays

### Arithmetic Operations

The operators `+`, `-`, `*`, `/` and `**` all act *elementwise* on arrays

```{code-cell} python3
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])
a + b
```

```{code-cell} python3
a * b
```

We can add a scalar to each element as follows

```{code-cell} python3
a + 10
```

Scalar multiplication is similar

```{code-cell} python3
a * 10
```

The two-dimensional arrays follow the same general rules

```{code-cell} python3
A = np.ones((2, 2))
B = np.ones((2, 2))
A + B
```

```{code-cell} python3
A + 10
```

```{code-cell} python3
A * B
```

In particular, `A * B` is *not* the matrix product, it is an element-wise product.

### Matrix Multiplication


With Anaconda's scientific Python package based around Python 3.5 and above,
one can use the `@` symbol for matrix multiplication, as follows:

```{code-cell} python3
A = np.ones((2, 2))
B = np.ones((2, 2))
A @ B
```

(For older versions of Python and NumPy you need to use the [np.dot](http://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html) function)

We can also use `@` to take the inner product of two flat arrays

```{code-cell} python3
A = np.array((1, 2))
B = np.array((10, 20))
A @ B
```

In fact, we can use `@` when one element is a Python list or tuple

```{code-cell} python3
A = np.array(((1, 2), (3, 4)))
A
```

```{code-cell} python3
A @ (0, 1)
```

Since we are post-multiplying, the tuple is treated as a column vector.

### Mutability and Copying Arrays

NumPy arrays are mutable data types, like Python lists.

In other words, their contents can be altered (mutated) in memory after initialization.

We already saw examples above.

Here's another example:

```{code-cell} python3
a = np.array([42, 44])
a
```

```{code-cell} python3
a[-1] = 0  # Change last element to 0
a
```

Mutability leads to the following behavior (which can be shocking to MATLAB programmers...)

```{code-cell} python3
a = np.random.randn(3)
a
```

```{code-cell} python3
b = a
b[0] = 0.0
a
```

What's happened is that we have changed `a` by changing `b`.

The name `b` is bound to `a` and becomes just another reference to the
array! 

Hence, it has equal rights to make changes to that array.

This is in fact the most sensible default behavior.

It means that we pass around only pointers to data, rather than making copies.

Making copies is expensive in terms of both speed and memory.

+++

#### Making Copies

It is of course possible to make `b` an independent copy of `a` when required.

This can be done using `np.copy`

```{code-cell} python3
a = np.random.randn(3)
a
```

```{code-cell} python3
b = np.copy(a)
b
```

Now `b` is an independent copy (called a *deep copy*)

```{code-cell} python3
b[:] = 1
b
```

```{code-cell} python3
a
```

Note that the change to `b` has not affected `a`.

## Additional Functionality

Let's look at some other useful things we can do with NumPy.

### Vectorized Functions

NumPy provides versions of the standard functions `log`, `exp`, `sin`, etc. that act *element-wise* on arrays

```{code-cell} python3
z = np.array([1, 2, 3])
np.sin(z)
```

This eliminates the need for explicit element-by-element loops such as

```{code-cell} python3
n = len(z)
y = np.empty(n)
for i in range(n):
    y[i] = np.sin(z[i])
```

Because they act element-wise on arrays, these functions are called *vectorized functions*.

In NumPy-speak, they are also called *ufuncs*, which stands for "universal functions".

As we saw above, the usual arithmetic operations (`+`, `*`, etc.) also
work element-wise, and combining these with the ufuncs gives a very large set of fast element-wise functions.

```{code-cell} python3
z
```

```{code-cell} python3
(1 / np.sqrt(2 * np.pi)) * np.exp(- 0.5 * z**2)
```

Not all user-defined functions will act element-wise.

For example, passing the function `f` defined below a NumPy array causes a `ValueError`

```{code-cell} python3
def f(x):
    return 1 if x > 0 else 0
```

The NumPy function `np.where` provides a vectorized alternative:

```{code-cell} python3
x = np.random.randn(4)
x
```

```{code-cell} python3
np.where(x > 0, 1, 0)  # Insert 1 if x > 0 true, otherwise 0
```

You can also use `np.vectorize` to vectorize a given function

```{code-cell} python3
f = np.vectorize(f)
f(x)                # Passing the same vector x as in the previous example
```

However, this approach doesn't always obtain the same speed as a more carefully crafted vectorized function.

### Comparisons

As a rule, comparisons on arrays are done element-wise

```{code-cell} python3
z = np.array([2, 3])
y = np.array([2, 3])
z == y
```

```{code-cell} python3
y[0] = 5
z == y
```

```{code-cell} python3
z != y
```

The situation is similar for `>`, `<`, `>=` and `<=`.

We can also do comparisons against scalars

```{code-cell} python3
z = np.linspace(0, 10, 5)
z
```

```{code-cell} python3
z > 3
```

This is particularly useful for *conditional extraction*

```{code-cell} python3
b = z > 3
b
```

```{code-cell} python3
z[b]
```

Of course we can---and frequently do---perform this in one step

```{code-cell} python3
z[z > 3]
```

### Sub-packages

NumPy provides some additional functionality related to scientific programming
through its sub-packages.

We've already seen how we can generate random variables using np.random

```{code-cell} python3
z = np.random.randn(10000)  # Generate standard normals
y = np.random.binomial(10, 0.5, size=1000)    # 1,000 draws from Bin(10, 0.5)
y.mean()
```

Another commonly used subpackage is np.linalg

```{code-cell} python3
A = np.array([[1, 2], [3, 4]])

np.linalg.det(A)           # Compute the determinant
```

```{code-cell} python3
np.linalg.inv(A)           # Compute the inverse
```


Much of this functionality is also available in [SciPy](http://www.scipy.org/), a collection of modules that are built on top of NumPy.

We'll cover the SciPy versions in more detail {doc}`soon <scipy>`.

For a comprehensive list of what's available in NumPy see [this documentation](https://docs.scipy.org/doc/numpy/reference/routines.html).


+++

## Ex. 1


Consider the polynomial expression

$$
p(x) = a_0 + a_1 x + a_2 x^2 + \cdots a_N x^N = \sum_{n=0}^N a_n x^n
$$

Now write a function that evaluates this expression but uses NumPy arrays and array operations for its computations, rather than any form of Python loop.

* Hint: Use `np.cumprod()`

+++

```
solutions below
solutions below
solutions below
solutions below
solutions below
solutions below
solutions below
solutions below
solutions below
solutions below
solutions below
solutions below
solutions below
solutions below
solutions below
solutions below
solutions below
solutions below
solutions below
solutions below
solutions below
```

## Solution to Ex. 1

```{code-cell} python3
def p(x, coef):
    X = np.ones_like(coef)
    X[1:] = x
    y = np.cumprod(X)   # y = [1, x, x**2,...]
    return coef @ y
```

Let's test it

```{code-cell} python3
x = 2
coef = np.linspace(2, 4, 3)
print(coef)
print(p(x, coef))
# For comparison
q = np.poly1d(np.flip(coef))
print(q(x))
```


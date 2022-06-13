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

## Overview

[SciPy](http://www.scipy.org) builds on top of NumPy to provide common tools for scientific programming such as

* [linear algebra](http://docs.scipy.org/doc/scipy/reference/linalg.html)
* [numerical integration](http://docs.scipy.org/doc/scipy/reference/integrate.html)
* [interpolation](http://docs.scipy.org/doc/scipy/reference/interpolate.html)
* [optimization](http://docs.scipy.org/doc/scipy/reference/optimize.html)
* [distributions and random number generation](http://docs.scipy.org/doc/scipy/reference/stats.html)
* [signal processing](http://docs.scipy.org/doc/scipy/reference/signal.html)
* etc., etc

+++

Many SciPy routines are thin wrappers around industry-standard Fortran libraries such as [LAPACK](https://en.wikipedia.org/wiki/LAPACK), [BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms), etc.

It's not really necessary to "learn" SciPy as a whole.

In this lecture, we aim only to highlight some useful parts of the package.

The functionality of SciPy is in its sub-packages

* `scipy.optimize`, `scipy.integrate`, `scipy.stats`, etc.

Let's explore some of the major sub-packages.

+++

## Statistics

The `scipy.stats` subpackage supplies

* numerous random variable objects (densities, cumulative distributions, random sampling, etc.)
* some estimation procedures
* some statistical tests

+++

### Random Variables and Distributions

Recall that `numpy.random` provides functions for generating random variables

```{code-cell} python3
import numpy as np
np.random.beta(5, 5, size=3)
```

This generates a draw from the distribution with the density function below when `a, b = 5, 5`

$$
f(x; a, b) = \frac{x^{(a - 1)} (1 - x)^{(b - 1)}}
    {\int_0^1 u^{(a - 1)} (1 - u)^{(b - 1)} du}
    \qquad (0 \leq x \leq 1)
$$

Sometimes we need access to the density itself, or the cdf, the quantiles, etc.

For this, we can use `scipy.stats`, which provides all of this functionality as well as random number generation in a single consistent interface.

Here's an example of usage

```{code-cell} ipython
%matplotlib inline
from scipy.stats import beta
import matplotlib.pyplot as plt

q = beta(5, 5)      # Beta(a, b), with a = b = 5
obs = q.rvs(2000)   # 2000 observations
grid = np.linspace(0.01, 0.99, 100)

fig, ax = plt.subplots()
ax.hist(obs, bins=40, density=True)
ax.plot(grid, q.pdf(grid), 'k-', linewidth=2)
plt.show()
```

The object `q` that represents the distribution has additional useful methods, including

```{code-cell} python3
q.cdf(0.4)      # Cumulative distribution function
```

```{code-cell} python3
q.ppf(0.8)      # Quantile (inverse cdf) function
```

```{code-cell} python3
q.mean()
```


## Roots and Fixed Points

A **root** or **zero** of a real function $f$ on $[a,b]$ is an $x \in [a, b]$ such that $f(x)=0$.

For example, if we plot the function

$$
f(x) = \sin(4 (x - 1/4)) + x + x^{20} - 1
$$

with $x \in [0,1]$ we get

```{code-cell} python3
f = lambda x: np.sin(4 * (x - 1/4)) + x + x**20 - 1
x = np.linspace(0, 1, 100)

fig, ax = plt.subplots()
ax.plot(x, f(x), label='$f(x)$')
ax.axhline(ls='--', c='k')
ax.set_xlabel('$x$', fontsize=12)
ax.set_ylabel('$f(x)$', fontsize=12)
ax.legend(fontsize=12)
plt.show()
```

The unique root is approximately 0.408.

Let's consider some numerical techniques for finding roots.

+++

### Bisection 

One of the most common algorithms for numerical root-finding is *bisection*.

To understand the idea, recall the well-known game where

* Player A thinks of a secret number between 1 and 100
* Player B asks if it's less than 50
    * If yes, B asks if it's less than 25
    * If no, B asks if it's less than 75

And so on.

This is bisection.

SciPy provides a bisection function.


```{code-cell} python3
from scipy.optimize import bisect

bisect(f, 0, 1)
```

### The Newton-Raphson Method 


Another very common root-finding algorithm is the [Newton-Raphson method](https://en.wikipedia.org/wiki/Newton%27s_method).

In SciPy this algorithm is implemented by `scipy.optimize.newton`.

Unlike bisection, the Newton-Raphson method uses local slope information in an attempt to increase the speed of convergence.

Let's investigate this using the same function $f$ defined above.

With a suitable initial condition for the search we get convergence:

```{code-cell} python3
from scipy.optimize import newton

newton(f, 0.2)   # Start the search at initial condition x = 0.2
```

But other initial conditions lead to failure of convergence:

```{code-cell} python3
newton(f, 0.7)   # Start the search at x = 0.7 instead
```

### Hybrid Methods

A general principle of numerical methods is as follows:

* If you have specific knowledge about a given problem, you might be able to exploit it to generate efficiency.
* If not, then the choice of algorithm involves a trade-off between speed and robustness.

In practice, most default algorithms for root-finding, optimization and fixed points use *hybrid* methods.

These methods typically combine a fast method with a robust method in the following manner:

1. Attempt to use a fast method
1. Check diagnostics
1. If diagnostics are bad, then switch to a more robust algorithm

In `scipy.optimize`, the function `brentq` is such a hybrid method and a good default

```{code-cell} python3
from scipy.optimize import brentq

brentq(f, 0, 1)
```

Here the correct solution is found and the speed is better than bisection:

```{code-cell} ipython
%timeit brentq(f, 0, 1)
```

```{code-cell} ipython
%timeit bisect(f, 0, 1)
```

## Optimization 

Most numerical packages provide only functions for *minimization*.

Maximization can be performed by recalling that the maximizer of a function $f$ on domain $D$ is
the minimizer of $-f$ on $D$.

Minimization is closely related to root-finding: For smooth functions, interior optima correspond to roots of the first derivative.

The speed/robustness trade-off described above is present with numerical optimization too.

Unless you have some prior information you can exploit, it's usually best to use hybrid methods.

For constrained, univariate (i.e., scalar) minimization, a good hybrid option is `fminbound`

```{code-cell} python3
from scipy.optimize import fminbound

fminbound(lambda x: x**2, -1, 2)  # Search in [-1, 2]
```

## Integration 

Most numerical integration methods work by computing the integral of an approximating polynomial.

The resulting error depends on how well the polynomial fits the integrand, which in turn depends on how "regular" the integrand is.

In SciPy, the relevant module for numerical integration is `scipy.integrate`.

A good default for univariate integration is `quad`

```{code-cell} python3
from scipy.integrate import quad

integral, error = quad(lambda x: x**2, 0, 1)
integral
```


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


# Numba

In addition to what's in Anaconda, this lecture will need the following libraries:

```{code-cell} ipython
!pip install quantecon
```

```{code-cell} ipython
%matplotlib inline
import numpy as np
import quantecon as qe
import matplotlib.pyplot as plt
```


## Overview

In an earlier lecture we learned about vectorization, which is one method to improve speed and efficiency in numerical work.

Vectorization involves sending array processing operations in batch to efficient low-level code.

However, vectorization has several weaknesses.

One is that it is highly memory-intensive when working with large amounts of data.

Another is that the set of algorithms that can be entirely vectorized is not universal.

In fact, for some algorithms, vectorization is ineffective.

Fortunately, a Python library called [Numba](http://numba.pydata.org/) solves many of these problems.

It does so through something called **just in time (JIT) compilation**.

The key idea is to compile functions to native machine code instructions on the fly.

When it succeeds, the compiled code is extremely fast.

This lecture introduces the main ideas.

+++

## Applications

As stated above, Numba's primary use is compiling functions to fast native
machine code during runtime.

### An Example

Let's consider a problem that is difficult to vectorize: generating the trajectory of a difference equation given an initial condition.

We will take the difference equation to be the quadratic map

$$
x_{t+1} = \alpha x_t (1 - x_t)
$$

In what follows we set

```{code-cell} python3
α = 4.0
```

Here's the plot of a typical trajectory, starting from $x_0 = 0.1$, with $t$ on the x-axis

```{code-cell} python3
def qm(x0, n):
    x = np.empty(n+1)
    x[0] = x0
    for t in range(n):
      x[t+1] = α * x[t] * (1 - x[t])
    return x

x = qm(0.1, 250)
fig, ax = plt.subplots()
ax.plot(x, 'b-', lw=2, alpha=0.8)
ax.set_xlabel('$t$', fontsize=12)
ax.set_ylabel('$x_{t}$', fontsize = 12)
plt.show()
```

To speed the function `qm` up using Numba, our first step is

```{code-cell} python3
from numba import jit

qm_numba = jit(qm)
```

The function `qm_numba` is a version of `qm` that is "targeted" for
JIT-compilation.

Let's time and compare identical function calls across these two versions, starting with the original function `qm`:

```{code-cell} python3
n = 10_000_000

qe.tic()
qm(0.1, int(n))
time1 = qe.toc()
```

Now let's try qm_numba

```{code-cell} python3
qe.tic()
qm_numba(0.1, int(n))
time2 = qe.toc()
```

This is already a massive speed gain.

In fact, the next time and all subsequent times it runs even faster as the function has been compiled and is in memory:

```{code-cell} python3
qe.tic()
qm_numba(0.1, int(n))
time3 = qe.toc()
```

```{code-cell} python3
time1 / time3  # Calculate speed gain
```

This kind of speed gain is huge relative to how simple and clear the implementation is.


### How and When it Works

Numba attempts to generate fast machine code using the infrastructure provided by the [LLVM Project](http://llvm.org/).

It does this by inferring type information on the fly.

The basic idea is this:

* Python is very flexible and hence we could call the function qm with many
  types.
    * e.g., `x0` could be a NumPy array or a list, `n` could be an integer or a float, etc.
* This makes it hard to *pre*-compile the function.
* However, when we do actually call the function, say by executing `qm(0.5, 10)`,
  the types of `x0` and `n` become clear.
* Moreover, the types of other variables in `qm` can be inferred once the input is known.
* So the strategy of Numba and other JIT compilers is to wait until this
  moment, and *then* compile the function.

That's why it is called "just-in-time" compilation.

Note that, if you make the call `qm(0.5, 10)` and then follow it with `qm(0.9, 20)`, compilation only takes place on the first call.

The compiled code is then cached and recycled as required.

## Decorators and "nopython" Mode

In the code above we created a JIT compiled version of `qm` via the call

```{code-cell} python3
qm_numba = jit(qm)
```

In practice this would typically be done using an alternative *decorator* syntax.

(We will explain all about decorators in a {doc}`later lecture <python_advanced_features>` but you can skip the details at this stage.)

Let's see how this is done.

### Decorator Notation

To target a function for JIT compilation we can put `@jit` before the function definition.

Here's what this looks like for `qm`

```{code-cell} python3
@jit
def qm(x0, n):
    x = np.empty(n+1)
    x[0] = x0
    for t in range(n):
        x[t+1] = α * x[t] * (1 - x[t])
    return x
```

This is equivalent to `qm = jit(qm)`.

The following now uses the jitted version:

```{code-cell} python3
qm(0.1, 10)
```

### Type Inference and "nopython" Mode

Clearly type inference is a key part of JIT compilation.

As you can imagine, inferring types is easier for simple Python objects (e.g., simple scalar data types such as floats and integers).

Numba also plays well with NumPy arrays.

In an ideal setting, Numba can infer all necessary type information.

This allows it to generate native machine code, without having to call the Python runtime environment.

In such a setting, Numba will be on par with machine code from low-level languages.

When Numba cannot infer all type information, some Python objects are given generic object status and execution falls back to the Python runtime.

When this happens, Numba provides only minor speed gains or none at all.

We generally prefer to force an error when this occurs, so we know effective
compilation is failing.

This is done by using either `@jit(nopython=True)` or, equivalently, `@njit` instead of `@jit`.

For example,

```{code-cell} python3
from numba import njit

@njit
def qm(x0, n):
    x = np.empty(n+1)
    x[0] = x0
    for t in range(n):
        x[t+1] = 4 * x[t] * (1 - x[t])
    return x
```


## Ex. 1

Previously we considered how to approximate $\pi$ by
Monte Carlo.

```{code-cell python3}
from random import uniform

def calculate_pi(n=1_000_000):
    count = 0
    for i in range(n):
        u, v = uniform(0, 1), uniform(0, 1)
        d = np.sqrt((u - 0.5)**2 + (v - 0.5)**2)
        if d < 0.5:
            count += 1

    area_estimate = count / n
    return area_estimate * 4  # dividing by radius**2
```

Make the code efficient using Numba.

Compare speed with and without Numba when the sample size is large.


+++


### Ex. 2

Write your own version of a one dimensional [kernel density
estimator](https://en.wikipedia.org/wiki/Kernel_density_estimation),
which estimates a density from a sample.

Write it as a class that takes the data $X$ and bandwidth
$h$ when initialized and provides a method $f$ such that

$$
f(x) = \frac{1}{hn} \sum_{i=1}^n
K \left( \frac{x-X_i}{h} \right)
$$

For $K$ use the Gaussian kernel ($K$ is the standard normal
density).

+++

Write the class so that the bandwidth defaults to Silverman’s rule (see
the “rule of thumb” discussion on [this
page](https://en.wikipedia.org/wiki/Kernel_density_estimation)). Test
the class you have written by going through the steps

1. simulate data $X_1, \ldots, X_n$ from distribution $\phi$
1. plot the kernel density estimate over a suitable range
1. plot the density of $\phi$ on the same figure

for distributions $\phi$ of the following types

- [beta
  distribution](https://en.wikipedia.org/wiki/Beta_distribution)
  with $\alpha = \beta = 2$
- [beta
  distribution](https://en.wikipedia.org/wiki/Beta_distribution)
  with $\alpha = 2$ and $\beta = 5$
- [beta
  distribution](https://en.wikipedia.org/wiki/Beta_distribution)
  with $\alpha = \beta = 0.5$

Use $n=500$.

Make a comment on your results. (Do you think this is a good estimator
of these distributions?)


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



+++

### Solution to Ex. 1

Since we want to compare speed, we'll avoid decorator syntax and keep a
non-jitted version:

```{code-cell} python3
from random import uniform

def calculate_pi(n=1_000_000):
    count = 0
    for i in range(n):
        u, v = uniform(0, 1), uniform(0, 1)
        d = np.sqrt((u - 0.5)**2 + (v - 0.5)**2)
        if d < 0.5:
            count += 1

    area_estimate = count / n
    return area_estimate * 4  # dividing by radius**2

calculate_pi_jitted = njit(calculate_pi)
```

Here's the speed comparison:

```{code-cell} python3
qe.tic()
calculate_pi()
time1 = qe.toc()
```

Now let's try qm_numba

```{code-cell} python3
calculate_pi_jitted()  # once to compile
qe.tic()
calculate_pi_jitted()  # and once to time
time2 = qe.toc()
```

Here's the time difference:

```{code-cell} python3
time1 / time2 
```


### Solution to Ex. 2

Here is one solution:

```{code-cell} ipython3
from scipy.stats import norm
from numba import vectorize, float64

@njit
def K(x):
    return (1/np.sqrt(2*np.pi)) * np.exp(-x**2 / 2)

def kde(x_data, h=None):

    if h is None:
        c = x_data.std()
        n = len(x_data)
        h = 1.06 * c * n**(-1/5)

    @vectorize([float64(float64)])
    def f(x):
        return K((x - x_data) / h).mean() * (1/h)

    return f
```



```{code-cell} ipython3
def plot_kde(ϕ, x_min=-0.2, x_max=1.2):
    x_data = ϕ.rvs(n)
    f = kde(x_data)

    x_grid = np.linspace(-0.2, 1.2, 100)
    fig, ax = plt.subplots()
    ax.plot(x_grid, f(x_grid), label="estimate")
    ax.plot(x_grid, ϕ.pdf(x_grid), label="true density")
    ax.legend()
    plt.show()
```

```{code-cell} ipython3
from scipy.stats import beta

n = 500
parameter_pairs= (2, 2), (2, 5), (0.5, 0.5)
for α, β in parameter_pairs:
    plot_kde(beta(α, β))
```

We see that the kernel density estimator is effective when the underlying
distribution is smooth but less so otherwise.


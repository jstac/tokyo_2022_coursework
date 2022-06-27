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


# Python Essentials

Now let's cover some core features of Python in a more systematic way.

+++

## Data Types

Programs typically track a range of data types.

For example, `1.5` is a floating point number, while `1` is an integer.

Programs need to distinguish between these two types for various reasons.

One is that they are stored in memory differently.

Another is that arithmetic operations are different

* For example, floating point arithmetic is implemented on most machines by a
  specialized Floating Point Unit (FPU).

In general, floats are more informative but arithmetic operations on integers
are faster and more accurate.

Python provides numerous other built-in Python data types, some of which we've already met

* strings, lists, etc.

Let's learn a bit more about them.

+++

### Primitive Data Types

One simple data type is **Boolean values**, which can be either `True` or `False`

```{code-cell} python3
x = True
x
```

We can check the type of any object in memory using the `type()` function.

```{code-cell} python3
type(x)
```

In the next line of code, the interpreter evaluates the expression on the right of = and binds y to this value

```{code-cell} python3
y = 100 < 10
y
```

```{code-cell} python3
type(y)
```

In arithmetic expressions, `True` is converted to `1` and `False` is converted `0`.

This is called **Boolean arithmetic** and is often useful in programming.

Here are some examples

```{code-cell} python3
x + y
```

```{code-cell} python3
x * y
```

```{code-cell} python3
True + True
```

```{code-cell} python3
bools = [True, True, False, True]  # List of Boolean values

sum(bools)
```

Complex numbers are another primitive data type in Python

```{code-cell} python3
x = complex(1, 2)
y = complex(2, 1)
print(x * y)

type(x)
```

### Containers

Python has several basic types for storing collections of (possibly heterogeneous) data.

We've already discussed lists.

A related data type is **tuples**, which are "immutable" lists

```{code-cell} python3
x = ('a', 'b')  # Parentheses instead of the square brackets
x = 'a', 'b'    # Or no brackets --- the meaning is identical
x
```

```{code-cell} python3
type(x)
```

In Python, an object is called **immutable** if, once created, the object cannot be changed.

Conversely, an object is **mutable** if it can still be altered after creation.

Python lists are mutable

```{code-cell} python3
x = [1, 2]
x[0] = 10
x
```

But tuples are not

```{code-cell} python3
---
tags: [raises-exception]
---
x = (1, 2)
x[0] = 10
```

We'll say more about the role of mutable and immutable data a bit later.


## Iterating

One of the most important tasks in computing is stepping through a
sequence of data and performing a given action.

One of Python's strengths is its simple, flexible interface to this kind of iteration via
the `for` loop.

### Looping over Different Objects

Many Python objects are "iterable", in the sense that they can be looped over.

To give an example, let's write the file us_cities.txt, which lists US cities and their population, to the present working directory.

```{code-cell} ipython
%%file us_cities.txt
new york: 8244910
los angeles: 3819702
chicago: 2707120
houston: 2145146
philadelphia: 1536471
phoenix: 1469471
san antonio: 1359758
san diego: 1326179
dallas: 1223229
```

Suppose that we want to make the information more readable, by capitalizing names and adding commas to mark thousands.

The program below reads the data in and makes the conversion:

```{code-cell} python3
data_file = open('us_cities.txt', 'r')
for line in data_file:
    city, population = line.split(':')         # Tuple unpacking
    city = city.title()                        # Capitalize city names
    population = f'{int(population):,}'        # Add commas to numbers
    print(city.ljust(15) + population)
data_file.close()
```

### Looping Techniques

Python provides some facilities to simplify looping 

One is `zip()`, which is used for stepping through pairs from two sequences.

For example, try running the following code

```{code-cell} python3
countries = ('Japan', 'Korea', 'China')
cities = ('Tokyo', 'Seoul', 'Beijing')
for country, city in zip(countries, cities):
    print(f'The capital of {country} is {city}')
```

Another useful one is `enumerate()` --- here's an example


```{code-cell} python3
countries = ('Japan', 'Korea', 'China')
cities = ('Tokyo', 'Seoul', 'Beijing')
for i, country in enumerate(countries):
    print(f'The capital of {country} is {cities[i]}')
```

### List Comprehensions


We can also simplify the code for generating the list of random draws considerably by using something called a *list comprehension*.

[List comprehensions](https://en.wikipedia.org/wiki/List_comprehension) are an elegant Python tool for creating lists.

Consider the following example, where the list comprehension is on the
right-hand side of the second line

```{code-cell} python3
animals = ['dog', 'cat', 'bird']
plurals = [animal + 's' for animal in animals]
plurals
```

Here's another example

```{code-cell} python3
range(8)
```

```{code-cell} python3
doubles = [2 * x for x in range(8)]
doubles
```

## Comparisons and Logical Operators

### Comparisons


Many different kinds of expressions evaluate to one of the Boolean values (i.e., `True` or `False`).

A common type is comparisons, such as

```{code-cell} python3
x, y = 1, 2
x < y
```

```{code-cell} python3
x > y
```

One of the nice features of Python is that we can *chain* inequalities

```{code-cell} python3
1 < 2 < 3
```

```{code-cell} python3
1 <= 2 <= 3
```

As we saw earlier, when testing for equality we use `==`

```{code-cell} python3
x = 1    # Assignment
x == 2   # Comparison
```

For "not equal" use `!=`

```{code-cell} python3
1 != 2
```

Note that when testing conditions, we can use **any** valid Python expression

```{code-cell} python3
x = 'yes' if 42 else 'no'
x
```

```{code-cell} python3
x = 'yes' if [] else 'no'
x
```

What's going on here?

The rule is:

* Expressions that evaluate to zero, empty sequences or containers (strings, lists, etc.) and `None` are all equivalent to `False`.
    * for example, `[]` and `()` are equivalent to `False` in an `if` clause
* All other values are equivalent to `True`.
    * for example, `42` is equivalent to `True` in an `if` clause

### Combining Expressions

```{index} single: Python; Logical Expressions
```

We can combine expressions using `and`, `or` and `not`.

These are the standard logical connectives (conjunction, disjunction and denial)

```{code-cell} python3
1 < 2 and 'f' in 'foo'
```

```{code-cell} python3
1 < 2 and 'g' in 'foo'
```

```{code-cell} python3
1 < 2 or 'g' in 'foo'
```

```{code-cell} python3
not True
```

```{code-cell} python3
not not True
```

Remember

* `P and Q` is `True` if both are `True`, else `False`
* `P or Q` is `False` if both are `False`, else `True`


## Keyword Arguments

We previously came across the statement `plt.plot(x, 'b-', label="white noise")`.

In this call to Matplotlib's `plot` function, notice that the last argument is passed in `name=argument` syntax.

This is called a *keyword argument*, with `label` being the keyword.

Non-keyword arguments are called *positional arguments*, since their meaning
is determined by order

* `plot(x, 'b-', label="white noise")` is different from `plot('b-', x, label="white noise")`

Keyword arguments are particularly useful when a function has a lot of arguments, in which case it's hard to remember the right order.

You can adopt keyword arguments in user-defined functions with no difficulty.

The next example illustrates the syntax

```{code-cell} python3
def f(x, a=1, b=1):
    return a + b * x
```

The keyword argument values we supplied in the definition of `f` become the default values

```{code-cell} python3
f(2)
```

They can be modified as follows

```{code-cell} python3
f(2, a=4, b=5)
```

## Coding Style and PEP8


To learn more about the Python programming philosophy type `import this` at the prompt.

Among other things, Python strongly favors consistency in programming style.

The standard style is set out in [PEP8](https://www.python.org/dev/peps/pep-0008/).

(Occasionally we'll deviate from PEP8 in these lectures to better match mathematical notation)


+++

## Exercises


+++

### Ex. 1  

Given two numeric lists or tuples `x_vals` and `y_vals` of equal length, compute
their inner product using `zip()`.


+++


### Ex. 2

In one line, count the number of even numbers in 0,...,99.

* Hint: `x % 2` returns 0 if `x` is even, 1 otherwise.


+++

### Ex. 3

Given `pairs = ((2, 5), (4, 2), (9, 8), (12, 10))`, count the number of pairs `(a, b)`
such that both `a` and `b` are even.


+++


### Ex. 4

Write a function that takes a string as an argument and returns the number of
capital letters in the string.


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


## Solutions

### Solution to Ex. 1

Here's one possible solution

```{code-cell} python3
x_vals = [1, 2, 3]
y_vals = [1, 1, 1]
sum([x * y for x, y in zip(x_vals, y_vals)])
```

This also works

```{code-cell} python3
sum(x * y for x, y in zip(x_vals, y_vals))
```

### Solution to Ex. 2

One solution is

```{code-cell} python3
sum([x % 2 == 0 for x in range(100)])
```

This also works:

```{code-cell} python3
sum(x % 2 == 0 for x in range(100))
```

Some less natural alternatives that nonetheless help to illustrate the
flexibility of list comprehensions are

```{code-cell} python3
len([x for x in range(100) if x % 2 == 0])
```

and

```{code-cell} python3
sum([1 for x in range(100) if x % 2 == 0])
```

### Solution to Ex. 3

Here's one possibility

```{code-cell} python3
pairs = ((2, 5), (4, 2), (9, 8), (12, 10))
sum([x % 2 == 0 and y % 2 == 0 for x, y in pairs])
```

### Solution to Ex. 4

Here's one solution:

```{code-cell} python3
def f(string):
    count = 0
    for letter in string:
        if letter == letter.upper() and letter.isalpha():
            count += 1
    return count

f('The Rain in Spain')
```

An alternative, more pythonic solution:

```{code-cell} python3
def count_uppercase_chars(s):
    return sum([c.isupper() for c in s])

count_uppercase_chars('The Rain in Spain')
```

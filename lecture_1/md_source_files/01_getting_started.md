---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: myst
      format_version: '1.3'
      jupytext_version: 1.13.7
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Setting up Your Python Environment

Aims

1. get a Python environment up and running
1. execute simple Python commands
1. run a sample program
1. install the code libraries that underpin these lectures

## Anaconda

The [core Python package](https://www.python.org/downloads/) is easy to install but *not* what you should choose for these lectures.

These lectures require the entire scientific programming ecosystem, which

* the core installation doesn't provide
* is painful to install one piece at a time.

Hence the best approach for our purposes is to install a Python distribution that contains

1. the core Python language **and**
1. compatible versions of the most popular scientific libraries.

The best such distribution is 

Anaconda is

* very popular
* cross-platform
* comprehensive
* completely unrelated to the [Nicki Minaj song of the same name](????)

Anaconda also comes with a great package management system to organize your code libraries.

+++

### Installing Anaconda

To install Anaconda, [download](https://www.anaconda.com/download/) the binary and follow the instructions.

Important points:

* Install the latest version!
* If you are asked during the installation process whether you'd like to make Anaconda your default Python installation, say yes.


+++

## Jupyter Notebooks 


[Jupyter](http://jupyter.org/) notebooks are one of the many possible ways to interact with Python and the scientific libraries.

They use  a *browser-based* interface to Python with

* The ability to write and execute Python commands.
* Formatted output in the browser, including tables, figures, animation, etc.
* The option to mix in formatted text and mathematical expressions.


While Jupyter isn't the only way to code in Python, it's great for when you wish to

* start coding in Python
* test new ideas or interact with small pieces of code
* share or collaborate scientific ideas with students or colleagues

+++

### Starting the Jupyter Notebook


* search for Jupyter in your applications menu, or
* open up a terminal and type `jupyter notebook`


+++

Let's now cover some basics:


* Running Cells
* Modal Editing
* Inserting Unicode 
* Tab Completion
* On-Line Help
* Adding rich text and LaTeX
* Installing Libraries
<!-- #endregion -->

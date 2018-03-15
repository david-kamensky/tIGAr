# tIGAr

A Python library for isogeometric analysis (IGA) using FEniCS.

## Dependencies
* Any meaningful usage requires [FEniCS](https://fenicsproject.org/) (version 2017.2) and its dependencies.
* Usage of the NURBS module requires [igakit](https://bitbucket.org/dalcinl/igakit).
* Compiling the API documentation requires [Sphinx](http://www.sphinx-doc.org/en/master/).
* The most convenient program for visualizing results is [ParaView](https://www.paraview.org/).

## Installation

Install all dependencies, clone the repository (or download and extract from an archive), and append the top-level directory of the repository (viz. the one with subdirectories `tIGAr`, `docs`, etc.) to the environment variable `PYTHONPATH`, e.g., by adding
```bash
export PYTHONPATH=/path/to/repository/:$PYTHONPATH
```
to your `~/.bashrc` file (and `source`-ing it).  To (optionally) build the API documentation, change directory to `docs` and type `make html`. The main documentation will then be in `./_build/html/index.html`, which can be opened with a web browser.  

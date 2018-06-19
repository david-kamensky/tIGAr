# tIGAr

A Python library for isogeometric analysis (IGA) using FEniCS.  The following article outlining the design of ``tIGAr`` is currently under review:
```
@article{Kamensky2018,
title = {{\sc tIGAr}: Automating isogeometric analysis with {FE}{\sc ni}{CS}},
journal = "Computer Methods in Applied Mechanics and Engineering",
author = "D. Kamensky and Y. Bazilevs",
year = {2018},
note = {Under review.}
}
```

## Dependencies
* Any meaningful usage requires [FEniCS](https://fenicsproject.org/) (version 2017.2) and its dependencies.
* [SciPy](https://www.scipy.org/) is required.  (SciPy is already included in FEniCS Docker distributions.)
* Usage of the NURBS module requires [igakit](https://bitbucket.org/dalcinl/igakit).
* Compiling the API documentation requires [Sphinx](http://www.sphinx-doc.org/en/master/).
* The most convenient program for visualizing results is [ParaView](https://www.paraview.org/).

## Installation

Install all dependencies, clone the repository (or download and extract from an archive), and append the top-level directory of the repository (viz. the one with subdirectories `tIGAr`, `docs`, etc.) to the environment variable `PYTHONPATH`, e.g., by adding
```bash
export PYTHONPATH=/path/to/repository/:$PYTHONPATH
```
to your `~/.bashrc` file (and `source`-ing it).  To (optionally) build the API documentation, change directory to `docs` and type `make html`. The main documentation will then be in `./_build/html/index.html`, which can be opened with a web browser.  

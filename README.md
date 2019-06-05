# tIGAr

A Python library for isogeometric analysis (IGA) using FEniCS.  The following article outlines the design of ``tIGAr``:
```
@article{Kamensky2019,
title = "{tIGAr}: Automating isogeometric analysis with {FEniCS}",
journal = "Computer Methods in Applied Mechanics and Engineering",
volume = "344",
pages = "477--498",
year = "2019",
issn = "0045-7825",
doi = "https://doi.org/10.1016/j.cma.2018.10.002",
author = "D. Kamensky and Y. Bazilevs"
}
```

## Dependencies
* Any meaningful usage requires [FEniCS](https://fenicsproject.org/) (version 2019.1) and its dependencies.
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

### Common installation issues
* `petsc4py.PETSc.Mat object has no attribute PtAP`: This is due to an old version of `petsc4py`.  Try installing the latest version via `pip3`.
* `ImportError: No module named dolfin`: This occurs when attempting to use `python` rather than `python3`.  FEniCS 2018.1 and newer no longer support Python 2.
* Errors due to old versions of FEniCS: Run `dolfin-version` in a terminal to check your version of FEniCS.  Note in particular that Ubuntu PPAs for the current stable version of FEniCS are only maintained for the most recent few Ubuntu releases.  Installing via the package manager on an old Ubuntu release may install an older version of FEniCS.

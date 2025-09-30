# DNA-origami-stability-prediction

![repo version](https://img.shields.io/badge/Version-v.%200.1.0-green)
![python version](https://img.shields.io/badge/python-3.9_|_3.10_|_3.11|_3.12-blue)
[![biorXiv](https://img.shields.io/badge/biorXiv-10.1101%2F2021.07.21.453083-red)](https://www.biorxiv.org/content/10.1101/2025.07.18.665506v1)
<!-- [![Static Badge](https://img.shields.io/badge/Data-Zenodo:_10.5281/8289605-54af7d)](https:///zenodo.org/records/8289605) -->

<!-- Title-->
<h1 id="Title">Predicting DNA origami stability in physiological media by machine learning</h1>

**Judith Zubia-Aranburu**<sup>1</sup>, **Andrea Gardin**<sup>1</sup>, **Lars Paffen**, **Matteo Tollemeto**, **Ane Alberdi**, **Maite Termenon**, **Francesca Grisoni**<sup>\*</sup>, **Tania Patiño Padial**<sup>\*</sup>\
<sup>1</sup>These authors contributed equally to this work.\
<sup>\*</sup>Corresponding authors: j.c.m.v.hest@tue.nl, f.grisoni@tue.nl, l.brunsveld@tue.nl.

<h2 id="disclaimer">Disclaimer</h2>

_The work has been submitted for peer-review and it might not represent the final version, the code and/or the content of the paper can be subjected to changes._

<!-- Abstract-->
<h2 id="abstract">Abstract</h2>
DNA origami nanostructures offer substantial potential as programmable, biocompatible platforms for drug delivery and diagnostics. However, their structural stability under physiological conditions remains a major barrier to practical applications. Stability assessment of DNA origami nanostructures has traditionally relied on image-based and empirical approaches, which are time-consuming and difficult to generalize across conditions. To address these limitations, we developed a machine learning approach for DNA origami stability prediction, based on measurable physicochemical parameters. Using dynamic light scattering (DLS) to quantify diffusion coefficients as a proxy for structural integrity, we characterized over 1400 DNA origami samples under varying physiologically relevant variables: temperature, incubation time, MgCl2 concentration, pH, and DNase I concentrations. The predictive performance of the model was confirmed using an independent set of samples under previously untested conditions. This data-driven approach offers a scalable and generalizable framework to guide the design of robust DNA nanostructures for biomedical applications.

![Figure 1](figures/fig1.png)

<!-- Content-->
<h2 id="content">Content</h2>

This repository contains the code used to apply the machine learning pipeline described in the main [paper](https://www.biorxiv.org/content/10.1101/2025.07.18.665506v1).

This repository is structured in the following way:
-   `/datasets/` : folder containing the datasets used to train/test/validate the models in our experiments.
-   `/experiments/` : folder containing the output data collected from the experiments described in the paper.
-   `/figures/` : folder containig high resolution figure as reported in the main paper.
-   `/script/` :  folder containig the script for running a replicate or a new experiment using the described ML set up.
-   `/src/origamiregressor/` : main folder containig the code modules defining the package.
-   `environment.yaml` : the environment file to create and install the package.
-   `pyproject.toml` : the setup file for installing the package.


<!-- Installation-->
<h2 id="content">Intallation</h2>

The package and all the needed dependencies can be installed with the provided `env.yaml` file. The installation was tested on Ubuntu 22.04.3.
```bash
conda env create -f environment.yaml
```

<!-- Running Experiments-->
<h2 id="content">Running Experiments</h2>

A quick tutorial on how to run the experiments, to reproduce and/or test the results, is given in the `./experiment/` folder.

<!-- How to cite-->
<h2 id="content">How to cite</h2>
You can currently cite our work from our preprint:

Predicting DNA origami stability in physiological media by machine learning.\
*Judith Zubia-Aranburu, Andrea Gardin, Lars Paffen, Matteo Tollemeto, Ane Alberdi, Maite Termenon, Francesca Grisoni, Tania Patiño Padial*
bioRxiv, 18 July, 2025. DOI: https://doi.org/10.1101/2025.07.18.665506


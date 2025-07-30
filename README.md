# DNA-origami-stability-prediction

![repo version](https://img.shields.io/badge/Version-v.%201.0.0-green)
![python version](https://img.shields.io/badge/python-3.9_|_3.10_|_3.11|_3.12-blue)
[![biorXiv](https://img.shields.io/badge/biorXiv-10.1101%2F2021.07.21.453083-red)](https://www.biorxiv.org/content/10.1101/2025.07.18.665506v1)
<!-- [![Static Badge](https://img.shields.io/badge/Data-Zenodo:_10.5281/8289605-54af7d)](https:///zenodo.org/records/8289605) -->

<!-- Title-->
<h1 id="Title">Predicting DNA origami stability in physiological media by machine learning</h1>

**Judith Zubia-Aranburu**<sup>1</sup>, **Andrea Gardin**<sup>1</sup>, **Lars Paffen**, **Matteo Tollemeto**, **Ane Alberdi**, **Maite Termenon**, **Francesca Grisoni**<sup>\*</sup>, **Tania Patiño Padial**<sup>\*</sup>\
<sup>1</sup>These authors contributed equally to this work.\
<sup>\*</sup>Corresponding authors: j.c.m.v.hest@tue.nl, f.grisoni@tue.nl, l.brunsveld@tue.nl.

<h2 id="disclaimer">Disclaimer</h2>

_This repo is not the final version an can be subjected to changes._

<!-- Abstract-->
<h2 id="abstract">Abstract</h2>
DNA origami nanostructures offer substantial potential as programmable, biocompatible platforms for drug delivery and diagnostics. However, their structural stability under physiological conditions remains a major barrier to practical applications. Stability assessment of DNA origami nanostructures has traditionally relied on image-based and empirical approaches, which are time-consuming and difficult to generalize across conditions. To address these limitations, we developed a machine learning approach for DNA origami stability prediction, based on measurable physicochemical parameters. Using dynamic light scattering (DLS) to quantify diffusion coefficients as a proxy for structural integrity, we characterized over 1400 DNA origami samples under varying physiologically relevant variables: temperature, incubation time, MgCl2 concentration, pH, and DNase I concentrations. The predictive performance of the model was confirmed using an independent set of samples under previously untested conditions. This data-driven approach offers a scalable and generalizable framework to guide the design of robust DNA nanostructures for biomedical applications.

![Figure 1](figures/fig1.png)

<!-- Content-->
<h2 id="content">Content</h2>

This repository contains the code used to apply the active machine learning pipeline described in the main [paper](https://www.biorxiv.org/content/10.1101/2025.07.18.665506v1).\
This repository is structured in the following way:
-   `/datasets/` : folder containing the datasets used to train/test/validate the models in our experiments.
-   `/figures/` : folder containig high resolution figure as reported in the main paper.
-   `/origamiregressor/` : main folder containig the `.py` modules defining the package.
-   `/script/` : 

# Hidden Variable Inverse Problems

Research project carried out with Arthur Conche at Université Paris Cité, motivated by applications in cryo-electron microscopy (Cryo-EM).

This project studies statistical estimation in inverse problems with latent variables. We investigate maximum likelihood estimation, gradient-based optimization, the Expectation-Maximization (EM) algorithm, regularization techniques, and applications to signal reconstruction from noisy transformed observations.

## Mathematical Report

The complete theoretical study is available in:

- report/doc-2-1.pdf

Topics include:

- Maximum Likelihood Estimation
- Expectation-Maximization (EM)
- Hidden Variables
- Inverse Problems
- Statistical Error Analysis
- Tikhonov Regularization
- Cryo-EM Motivation

## Software Design

A modular Python implementation is currently under development.

Planned components:

- Generic latent-variable inverse problem framework
- 1D translation model
- 2D rotation-projection model
- EM and GEM algorithms
- Numerical experiments

## Repository Structure
core/       # EM and optimization algorithms
models/     # Translation and rotation-projection models
utils/      # Metrics and utilities
figures/    # Experimental results
rapport/    # Report and presentation slides
## Future Work
  - Convergence analysis
  - Continuous latent-variable models
  - Advanced regularization techniques
  - Large-scale Cryo-EM applications

## Acknowledgements

## Acknowledgements

This project was carried out jointly with Arthur Conche as part of a Master's research project. Part of the software implementation was developed in collaboration with him.

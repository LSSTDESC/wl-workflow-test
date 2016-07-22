# WL Workflow concept

From the DESC Hack Day 2016-07-22

Use the SLAC Workflow Engine to execute the initial WL pipeline for cosmological parameter estimation from cosmic shear and angular galaxy clustering.

This pipeline takes as input image data or simulated image data, performs selections and null tests, computes summary statistics, and samples from the cosmological parameter posterior distribution.

## Pipeline steps

**Bolded entries** have initial implementations. All other steps are *no operation*.

[Data or simulation steps go here]

- DM catalog
- Selection
- Null tests on catalog
- **Tomographic binning**
- Photo-z characterization
- **dN/dz inference**
- **2PCF estimator**
- Null tests on correlation functions
- Covariance model
- TJPCosmo




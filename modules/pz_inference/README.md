# Installing BPZ

**Copied from WFIRST WPS project - Authors: W. A. Dawson, S. Schmidt**

The BPZ files have been download from Dan Coe at http://www.stsci.edu/~dcoe/BPZ/ and added to this repo. So you shouldn't need to download anything. All you need to do is add the following lines to your bash profile (or modified slightly if using tcsh):

    # Add configurations for BPZ
    export BPZPATH=$HOME/wl-workflow-test/modules/pz_inference/bpz-1.99.3
    export PYTHONPATH=$PYTHONPATH:$BPZPATH
    alias bpz="python $BPZPATH/bpz.py"
    export NUMERIX=numpy

where the exact BPZPATH will vary depending on your specific path to the `wfirst_photoz` repo.

# Introduction

Most of the interface with BPZ is done through the `.cat`, `.columns`, and `.pars` files.

## The `.cat` File

The `.cat` file is an ascii catalog that contains rows of objects and columns with `mag_i` and `mag_err_i` columns for each ith filter.

## The `.columns` File

The `.columns` file is an ascii file that indexes the `.cat` file column numbers (1 based index), and also specifies various conventions and potential added error terms. E.g.:

    #Filter        columns  cal(AB or Vega,optional)  zp_error  zp_offset
    2015LSSTu      7,17	    AB     	  		  0.005	    0.0
    2015LSSTg      8,18	    AB			  0.005	    0.0
    Z_S 	       	2
    #ID		1
    M_0		9
    OTHER		1,2,3,4,5,6

* columns is name of columns file (name of filter as specified in the FILTER directory but with out the .res extension, columns mag,mag_err, AB or Vega, bp_error, zp_offset)
* cal tells whether the corresponding magnitudes are in AB or Vega.
* Sam usually deals zp_error of 0.005 since that is the LSST zero point systematic error budget value.
* zp_offset usually set to 0
* M_0 apparent magnitude prior (magnitude as a function of redshift), specify the column to use as the magnitude to use with the corresponding PRIOR in the .pars files. Note that this
* Z_S is the column of the spec z (doesn't really do anything for default BPZ run)
* OTHER columns to append to the results file.

## The `.pars` File

The `.pars` file is similar to the SExtractor parameter file. Here is a summary of some of the cards:
* OUTPUT is the name of the z_b single point estimates
* SPECTRA the .list file of the SED spectra files, that will be saved as .AB files which are the spectra at a given redshift convolved with each of the filters
* PRIOR name of the .py file that specifies the parametrized prior
* DZ is the delta z spacing that it checks
* ZMIN, ZMAX is the min and max redshifts to check for
* NEW_AB, if set to no then it will save time and not recompute .AB convolution files.  If you change Madau, set NEW_AB to yes at least once or you will get wrong answers!
* MADAU, we want this set to NO because we don't have a model for extinction due to neutral hydrogen in our simulated Universe of blends
* ZC, FC add a cluster redshift delta function prior to the prior
* INTERP if set to N it will add N linearly interpolated SEDs between each pair of templates, if 0 no interpolation is done it just uses the templates you specified.
* ODDS 0.68, this parameter changes the output values of Z_B_MIN and Z_B_MAX to be the redshift interval enclosing that fraction of flux (e.g. ODDS .95 would give the values of Z_B_MIN and Z_B_MAX that enclose 95% of the pdf.
* PROBS_LITE set it to the name of the file that will contain the 1D marginal p(z) distributions
* CONVOLVE_P set to "yes", this adds a small amount of smoothing to the final p(z) to avoid single value peaks 
* P_MIN think about setting to 1e-4 (this parameter sets p(z) to 0.0 if p(z)< P_MIN)

## Output
Will output a `.bpz` file
* Z_B is the single point best estimate of redshift at the peak of the p(z) curve.
* T_B is the best fit type
* ODDS is a quality measure that calculates the fraction of p(z) within 0.06(1+zb) of the zb value (i.e. how "peaked" the p(z) distribution is)
* Z_ML and T_ML is just the best values but with applying any priors

## Location of Files

* The .cat, .pars, and .columns files can be in any directory.
* The .priors files must be put in $BPZPATH.
* The SPECTRA .list files are in the $BPZPATH/SED path.


# Running BPZ

    bpz name.cat -P name.pars
    python $BPZPATH/bpzfinalize.py name
    python $BPZPATH/plots/webpage.py name

where the name.cat file is the one that contains mag and mag_errs columns, with a row for each galaxy, and the name.pars is the file.

Note that the first time you run BPZ with new filter curves and SED templates it will take a while to generate the convolution of these for various redshifts (generating .AB files), however after this is done once it should use the same .AB files in the future.
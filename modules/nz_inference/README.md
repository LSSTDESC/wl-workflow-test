#Setup

##The mcmc directory

Download the CHIPPR .py scripts into a directory "mcmc".

##The tests directory

In the directory above "mcmc" make a directory "tests".  Within "tests", there must be three types of items: a file "tests-mcmc.txt" containing the name(s) of the file(s) containing input parameters for the case(s) you would like to run (each on its own line), the .txt file(s) containing input parameters, and a directory (for each test case) with a name shared with its corresponding .txt file.

##The input parameters file

Each line should contain the name of the input parameter, a single space, and then the value associated with that parameter.  The possible input parameters (all optional) are detailed below:

name: the title of the test case, with no spaces
priormean: series of floating point values separated by spaces, indicating the mean of the prior distribution
priorcov: series of floating point values separated by spaces, indicating the row-wise elements of the covariance matrix of the prior distribution
inits: "gs" to initialize walkers with Gaussian samples around a sample from the prior distribution, "ps" to initialize walkers with samples from the prior distribution, or "gm" to initialize walkers with Gaussian samples around the mean of the prior distribution
miniters: integer equal to the log-base-10 of the number of iterations defining a sub-run
factor: integer equal to number of post-burn sub-runs to save
thinto: 0 to not thin the chains, 1 to thin the chains by a factor of 10
mode: "bins" to calculate autocorrelation times per bin, "walkers" to calculate autocorrelation times per walker
plotonly: 0 to run MCMC and then make plots, 1 to make plots based on existing samples

##The test case directory

Each test case directory must contain a "data" directory.

##The data directory

The "data" directory must contain at least one file entitled "logdata.csv".  It must contain the following: the endpoints of redshift bins, the log interim prior, and the log interim posterior probabilities, in that order.  Each item must be on its own line with spaces separating the elements.

##Example

name TestCase
inits gs
miniters 3
thinto 1
factor 10
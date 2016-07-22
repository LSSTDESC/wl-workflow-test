"""
util-mcmc module defines handy tools for MCMC
"""

import sys
import numpy as np
import statistics
import cPickle as cpkl
import scipy as sp

def lrange(l):
    """
    lrange(l) makes a range based on the length of a list or array l
    """
    return xrange(len(l))

def safelog(xarr):
    """
    safelog takes log of array with zeroes
    """
    shape = np.shape(xarr)
    flat = xarr.flatten()
    logged = np.log(np.array([max(x,sys.float_info.epsilon) for x in flat]))
    return logged.reshape(shape)

def normed(x,scale):
    """
    normed takes a numpy array and returns a normalized version of it that integrates to 1
    """
    x = np.array(x)
    scale = np.array(scale)
    norm = x/np.dot(x,scale)
    return norm

class mvn(object):
    """
    mvn object is multivariate normal distribution, to be used in data generation and prior to emcee
    """
    def __init__(self, mean, cov):
        """input multidimensional mean and covariance matrix as numpy arrays"""
        self.dims = len(mean)
        self.mean = mean
        self.cov = cov
        self.icov = np.linalg.pinv(self.cov, rcond=sys.float_info.epsilon)
        (self.logdetsign, self.logdet) = np.linalg.slogdet(self.cov)

    def logpdf(self, x):
        """log probabilities"""
        delta = x - self.mean
        c = np.dot(delta, np.dot(self.icov, delta))
        prob = -0.5 * c
        return prob

    def sample_ps(self, W):
        """W samples directly from distribution"""
        outsamp = np.random.multivariate_normal(self.mean, self.cov, W)
        return (outsamp, self.mean)

    def sample_gm(self,W):
        """W samples around mean of distribution"""
        outsamp = [self.mean+np.random.randn(self.dims) for w in range(0,W)]
        return (outsamp, self.mean)

    def sample_gs(self, W):
        """W samples from a single sample from distribution"""
        rando = np.random.multivariate_normal(self.mean, self.cov)
        #outsamp = [rando + np.sqrt(rando)*np.random.randn(self.dims) for w in range(0,W)]
        outsamp = [np.random.multivariate_normal(rando,self.cov) for w in range(0,W)]
        return (outsamp, rando)

class post(object):
    """
    post object is posterior distribution we wish to sample
    """
    def __init__(self,idist,xvals,yprobs,interim):
        """data are logged posteriors (ngals*nbins), idist is mvn object"""
        self.prior = idist
        #self.priormean = idist.mean
        self.interim = interim
        self.xgrid = np.array(xvals)
        self.difs = self.xgrid[1:]-self.xgrid[:-1]#np.array([self.xgrid[k+1]-self.xgrid[k] for k in self.dims])
        self.lndifs = np.log(self.difs)#np.array([m.log(max(self.difs[k],sys.float_info.epsilon)) for k in self.dims])
        self.postprobs = yprobs
        self.constterm = self.lndifs-self.interim#self.priormean
        self.lnprob_ext = post_lnprob

    def priorprob(self,theta):
        """this is proportional to log prior probability"""
        return self.prior.logpdf(theta)

    def lnlike(self,theta):
        """specific to N(z) problem"""
        #return self.lnprob(theta)-self.priorprob(theta)
        constterms = theta+self.constterm
        sumterm = -1.*np.dot(np.exp(theta),self.difs)
        for j in lrange(self.postprobs):
            logterm = np.log(np.sum(np.exp(self.postprobs[j]+constterms)))
            sumterm += logterm
        return sumterm

    def mlnlike(self,theta):
        return -1.*self.lnlike(theta)

    # speed this up some more with matrix magic?
    def lnprob(self,theta):
        """calculate log posterior probability"""
#         constterms = theta+self.constterm
#         sumterm = self.priorprob(theta)-np.dot(np.exp(theta),self.difs)#this should sufficiently penalize poor samples but somehow fails on large datasets
#         for j in lrange(self.postprobs):
#             #logterm = sp.misc.logsumexp(self.postprobs[j]+constterms)#shockingly slower!
#             #logterm = np.logaddexp(self.postprobs[j]+constterms)#only works for two terms
#             logterm = np.log(np.sum(np.exp(self.postprobs[j]+constterms)))
#             sumterm += logterm
#         return sumterm
        return self.lnlike(theta)+self.priorprob(theta)

def post_lnprob(theta, other_self):
    ret = other_self.lnprob(theta)
    return ret

class path(object):
    """
    path object takes templates of path style and variables for it and makes os.path objects from them
    """
    def __init__(self, path_template, filled = None):
        self.path_template = path_template
        if filled is None:
            self.filled = {}
        else:
            self.filled = filled

    def construct(self, **args):
        """actually constructs the final path, as a string.  Optionally takes in any missing parameters"""
        nfilled = self.filled.copy()
        nfilled.update(args)
        return self.path_template.format(**nfilled)

    def fill(self, **args):
        """fills any number of missing parameters, returns new object"""
        dct = self.filled.copy()
        dct.update(args)
        return path(self.path_template, dct)

class tnorm(object):
    """truncated normal distribution object"""
    def __init__(self,mu,sig,ends):
        self.mu = mu
        self.sig = sig
        (self.min,self.max) = ends
        self.lo = self.loc(self.min)
        self.hi = self.loc(self.max)

    def loc(self,z):
        return (z-self.mu)/self.sig

    def phi(self,z):
        x = z/np.sqrt(2)
        term = sp.special.erf(x)
        return (1.+term)/2.

    def norm(self):
        return max(sys.float_info.epsilon,self.phi(self.hi)-self.phi(self.lo))

    def pdf(self,z):
        x = self.loc(z)
        pdf = sp.stats.norm.pdf(x)
        return pdf/(self.sig*self.norm())

    def cdf(self,z):
        x = self.loc(z)
        cdf = self.phi(x)-self.phi(self.lo)
        return cdf/self.norm()

    def rvs(self,J):
        func = sp.stats.truncnorm(self.lo,self.hi,loc=self.mu,scale=self.sig)
        return func.rvs(size=J)

class gmix(object):
    """
    gmix object takes a numpy array of Gaussian parameters and enables computation of PDF
    """
    def __init__(self,inarr,bounds):

        self.minZ,self.maxZ = bounds
        self.comps = inarr
        self.ncomps = len(self.comps)

        self.weights = np.transpose(self.comps)[2]
#         mincomps = [(self.minZ-comp[0])/comp[1] for comp in self.comps]
#         maxcomps = [(self.maxZ-comp[0])/comp[1] for comp in self.comps]
        self.comps = [tnorm(comp[0],comp[1],(self.minZ,self.maxZ)) for comp in self.comps]#[sp.stats.truncnorm(mincomps[c],maxcomps[c],loc=self.comps[c][0],scale=self.comps[c][1]) for c in lrange(self.comps)]
#         self.comps = [tnorm(comp[0],comp[1],(self.minZ,self.maxZ)) for comp in self.comps]
#         self.weights = np.array([self.calccdf(c,self.minZ,self.maxZ) for c in lrange(self.comps)])

    def pdfs(self,zs):
        out = np.array([self.weights[c]*np.array([self.comps[c].pdf(z) for z in zs]) for c in xrange(self.ncomps)])
        return out

    def pdf(self,zs):
        return np.sum(self.pdfs(zs),axis=0)

    def cdfs(self,zs):
        out = np.array([self.weights[c]*np.array([self.comps[c].cdf(z) for z in zs]) for c in xrange(self.ncomps)])
        return out

    def cdf(self,zs):
        return np.sum(self.cdfs(zs),axis=0)

    def binned(self,zs):
        thing = self.cdf(zs)
        return thing[1:]-thing[:-1]

    def sample(self,N):
        choices = [0]*self.ncomps
        for j in xrange(N):
            choices[choice(xrange(self.ncomps), self.weights)] += 1
        samps = np.array([])
        for c in xrange(self.ncomps):
            j = choices[c]
            Zs = self.comps[c].rvs(j)
            samps = np.concatenate((samps,Zs))
        return np.array(samps)

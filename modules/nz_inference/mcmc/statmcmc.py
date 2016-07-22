"""
stat-mcmc module calculates intermediate statistics for monitoring state
"""

import statistics
import numpy as np
import cPickle as cpkl
import os
import scipy as sp
import csv

import utilmcmc as um

class calcstats(object):
    """
    object class to set up and calculate summary statistics, unite stats for each output
    """
    def __init__(self, meta):
        self.meta = meta
    def update(self, ydata):
        if self.meta.plotonly == False:
            stats = self.compute(ydata)
            self.meta.key.add_stats(self.meta.topdir, self.name, stats)

# # statistics involving both log posterior probabilities and parameter values
# class stat_both(calcstats):
#     """
#     calculates statistics that require both posterior probabilities and parameter values: log likelihood ratio and MAP parameter values
#     """
#     def __init__(self,meta):
#         calcstats.__init__(self,meta)

#         self.name = 'both'

#         self.ll_stk = self.meta.postdist.lnlike(self.meta.logstkNz)
# #         self.ll_map = self.meta.postdist.lnlike(self.meta.logmapNz)
# #         self.ll_exp = self.meta.postdist.lnlike(self.meta.logexpNz)
#         self.ll_int = self.meta.postdist.lnlike(self.meta.logintNz)
#         self.ll_mml = self.meta.postdist.lnlike(self.meta.logmmlNz)
#         self.ll_smp = []
# #         self.mapvals,self.maps = [],[]

# #         self.llr_stk,self.llr_map,
#         self.llr_stk,self.llr_int,self.llr_mml = [],[],[]

#         outdict = {'ll_stk': self.ll_stk,
# #                   'll_map': self.ll_map,
# #                   'll_exp': self.ll_exp,
#                   'll_int': self.ll_int,
#                   'll_mml': self.ll_mml,
#                   'll_smp': self.ll_smp,
#                   'llr_stk': np.array(self.llr_stk),
# #                   'llr_map': np.array(self.llr_map),
# #                   'llr_exp': np.array(self.llr_exp),
#                   'llr_int': np.array(self.llr_int),
#                   'llr_mml': np.array(self.llr_mml)
#                    }
#         if self.meta.plotonly == False:
#             with open(os.path.join(self.meta.topdir,'stat_both.p'),'wb') as statboth:
#                 cpkl.dump(outdict,statboth)

#     def compute(self,ydata):

#         self.probs = ydata['probs']
#         self.chains = ydata['chains']

# #         where = np.unravel_index(np.argmax(self.probs),(self.meta.nwalkers,self.meta.ntimes))
# #         self.mapvals.append(self.chains[where])
# #         self.maps.append(self.probs[where])

#         self.llr_stk = self.calclr(self.llr_stk,self.ll_stk)
# #         self.llr_map = self.calclr(self.llr_map,self.ll_map)
# #         self.llr_exp = self.calclr(self.llr_exp,self.ll_exp)
#         self.llr_mml = self.calclr(self.llr_mml,self.ll_mml)

#         if self.meta.logtruNz is not None:
#             self.calclr(self.ll_smp,0.)

#         with open(os.path.join(self.meta.topdir,'stat_both.p'),'rb') as indict:
#             outdict = cpkl.load(indict)

#         outdict['llr_stk'] = np.array(self.llr_stk)
# #         outdict['llr_map'] = np.array(self.llr_map)
# #         outdict['llr_exp'] = np.array(self.llr_exp)
#         outdict['llr_mml'] = np.array(self.llr_mml)
#         outdict['ll_smp'] = np.array(self.ll_smp).flatten()/2.
# #         outdict['mapvals'] = np.array(self.mapvals)
# #         outdict['maps'] = np.array(self.maps)

#         with open(os.path.join(self.meta.topdir,'stat_both.p'),'wb') as statboth:
#             cpkl.dump(outdict,statboth)
#         return

#     # likelihood ratio test
#     def calclr(self,var,ll):

#         for w in xrange(self.meta.nwalkers):
#             for x in xrange(self.meta.ntimes):
#                 ll_smp = self.probs[w][x]-self.meta.postdist.priorprob(self.chains[w][x])
#                 var.append(2.*(ll_smp-ll))
# #                 self.llr_stk.append(2.*ll_smp-2.*self.ll_stk)
# #                 self.llr_map.append(2.*ll_smp-2.*self.ll_map)
# #                 self.llr_exp.append(2.*ll_smp-2.*self.ll_exp)
#         return(var)

class stat_chains(calcstats):
    """
    calculates statistics that need parameter values: variance, chi^2, KLD; statistics involving parameter values
    """
    def __init__(self, meta):
        calcstats.__init__(self, meta)

        self.name = 'chains'

        self.var_ls = []
        self.var_s = []
#         self.vslogstk = None
#         self.vslogmap = None
# #         self.vslogexp = None
#         self.vsstk = None
#         self.vsmap = None
# #         self.vsexp = None

        self.chi_ls = []
        self.chi_s = []
#         self.cslogstk = None
#         self.cslogmap = None
# #         self.cslogexp = None
#         self.csstk = None
#         self.csmap = None
# #         self.csexp = None

        self.kl_stkvtru,self.kl_truvstk = None,None
        self.kl_mapvtru,self.kl_truvmap = None,None
        self.kl_expvtru,self.kl_truvexp = None,None
        self.kl_intvtru,self.kl_truvint = float('inf'),float('inf')
        self.kl_mmlvtru,self.kl_truvmml = float('inf'),float('inf')
        self.kl_smpvtru,self.kl_truvsmp = float('inf'),float('inf')

        if self.meta.logtruNz is not None:
#             vslogstk = self.meta.logstkNz-self.meta.logtruNz
#             self.vslogstk = np.dot(vslogstk,vslogstk)
#             vslogmap = self.meta.logmapNz-self.meta.logtruNz
#             self.vslogmap = np.dot(vslogmap,vslogmap)
# #             vslogexp = self.meta.logexpNz-self.meta.logtruNz
# #             self.vslogexp = np.dot(vslogexp,vslogexp)

#             self.cslogstk = np.average((self.meta.logstkNz-self.meta.logtruNz)**2)
#             self.cslogmap = np.average((self.meta.logmapNz-self.meta.logtruNz)**2)
# #             self.cslogexp = np.average((self.meta.logexpNz-self.meta.logtruNz)**2)

            self.kl_stkvtru,self.kl_truvstk = calckl(self.meta.bindifs,self.meta.logstkNz,self.meta.logtruNz)
            self.kl_mapvtru,self.kl_truvmap = calckl(self.meta.bindifs,self.meta.logmapNz,self.meta.logtruNz)
            self.kl_expvtru,self.kl_truvexp = calckl(self.meta.bindifs,self.meta.logexpNz,self.meta.logtruNz)
            self.kl_intvtru,self.kl_truvint = calckl(self.meta.bindifs,self.meta.logintNz,self.meta.logtruNz)
            self.kl_mmlvtru,self.kl_truvmml = calckl(self.meta.bindifs,self.meta.logmmlNz,self.meta.logtruNz)
            self.kl_smpvtru,self.kl_truvsmp = [],[]

#         if self.meta.truNz is not None:
#             vsstk = meta.stkNz-meta.truNz
#             self.vsstk = np.dot(vsstk,vsstk)
#             vsmap = meta.mapNz-meta.truNz
#             self.vsmap = np.dot(vsmap,vsmap)
# #             vsexp = meta.expNz-meta.truNz
# #             self.vsexp = np.dot(vsexp,vsexp)

#             self.csstk = np.average((self.meta.stkNz-self.meta.truNz)**2)
#             self.csmap = np.average((self.meta.mapNz-self.meta.truNz)**2)
#             self.csexp = np.average((self.meta.expNz-self.meta.truNz)**2)

        outdict = {#'vslogstk': self.vslogstk,
#                    'vsstk': self.vsstk,
#                    'vslogmap': self.vslogmap,
#                    'vsmap': self.vsmap,
# #                    'vslogexp': self.vslogexp,
# #                    'vsexp': self.vsexp,
#                    'cslogstk': self.cslogstk,
#                    'csstk': self.csstk,
#                    'cslogmap': self.cslogmap,
#                    'csmap': self.csmap,
# #                    'cslogexp': self.cslogexp,
# #                    'csexp': self.csexp,
                   'kl_stkvtru': self.kl_stkvtru,
                   'kl_mapvtru': self.kl_mapvtru,
                   'kl_expvtru': self.kl_expvtru,
                   'kl_smpvtru': self.kl_smpvtru,
                   'kl_intvtru': self.kl_intvtru,
                   'kl_mmlvtru': self.kl_mmlvtru,
                   'kl_truvstk': self.kl_truvstk,
                   'kl_truvmap': self.kl_truvmap,
                   'kl_truvexp': self.kl_truvexp,
                   'kl_truvsmp': self.kl_truvsmp,
                   'kl_truvint': self.kl_truvint,
                   'kl_truvmml': self.kl_truvmml
              }
        if self.meta.plotonly == False:
            with open(os.path.join(self.meta.topdir,'stat_chains.p'),'wb') as statchains:
                cpkl.dump(outdict,statchains)

    def compute(self, ydata):#ntimes*nwalkers*nbins

        self.ydata = ydata
        self.eydata = np.exp(self.ydata)

#         print('about to write samples to file: '+str(self.meta.key.burnin))
#         if self.meta.key.burnin == False:

        with open(self.meta.samples,'ab') as csvfile:
            out = csv.writer(csvfile,delimiter=' ')
            for w in xrange(self.meta.nwalkers):
                out.writerows(self.ydata[w])#[[x for x in row] for row in self.ydata])
#             print(str(self.meta.key.burnin)+'wrote samples to file')
#         else:
#             print('not writing samples to file because still burning in: '+str(self.meta.key.burnin))

        y = np.swapaxes(self.ydata.T,0,1).T#nwalkers*nbins*ntimes
        ey = np.swapaxes(self.eydata.T,0,1).T#np.exp(y)

        if self.meta.logtruNz is None:
            my = np.array([[[sum(by)/len(by)]*self.meta.ntimes for by in wy] for wy in y])#nwalkers*nbins*ntimes
            mey = np.array([[[sum(bey)/len(bey)]*self.meta.ntimes for bey in wey] for wey in ey])#nwalkers*nbins*ntimes
        else:
            my = np.array([[[k]*self.meta.ntimes for k in self.meta.logtruNz]]*self.meta.nwalkers)#nwalkers*nbins*ntimes
            mey = np.array([[[k]*self.meta.ntimes for k in self.meta.truNz]]*self.meta.nwalkers)#nwalkers*nbins*ntimes

        self.sy = np.swapaxes((y-my),1,2)#nwalkers*ntimes*nbins to #nwalkers*nbins*ntimes
        self.sey = np.swapaxes((ey-mey),1,2)

        self.var_ls = self.calcvar(self.var_ls,self.sy)
        self.var_s = self.calcvar(self.var_s,self.sey)
        self.chi_ls = self.calcchi(self.chi_ls,self.sy,self.ydata)
        self.chi_s = self.calcchi(self.chi_s,self.sey,self.eydata)

        with open(os.path.join(self.meta.topdir,'stat_chains.p'),'rb') as indict:
            outdict = cpkl.load(indict)

#         outdict['kl_smpvtru'] = np.array(self.kl_smpvtru)
#         outdict['kl_truvsmp'] = np.array(self.kl_truvsmp)
        #outdict['tot_var_ls'] = self.tot_var_ls
        #outdict['tot_var_s'] = self.tot_var_s
        outdict['var_ls'] = self.var_ls
        outdict['var_s'] = self.var_s
        #outdict['tot_chi_ls'] = self.tot_chi_ls
        #outdict['tot_chi_s'] = self.tot_chi_s
        outdict['chi_ls'] = self.chi_ls
        outdict['chi_s'] = self.chi_s

        with open(os.path.join(self.meta.topdir,'stat_chains.p'),'wb') as indict:
            cpkl.dump(outdict,indict)
            #print('after addition'+str(outdict))
#         return { 'vslogstk': self.vslogstk,
#                'vsstk': self.vsstk,
#                'vslogmap': self.vslogmap,
#                'vsmap': self.vsmap,
#                'vslogexp': self.vslogexp,
#                'vsexp': self.vsexp,
#                'tot_var_ls': self.tot_var_ls,
#                'tot_var_s': self.tot_var_s,
#                'var_ls': self.var_ls,
#                'var_s': self.var_s
#               }

    def calcvar(self,var,s):
        """variance of samples"""
        ans = np.average([[np.dot(s[w][i],s[w][i]) for i in xrange(len(s[w]))] for w in xrange(len(s))])
        var.append(ans)
#         var_ls = np.average([[np.dot(self.sy[w][i],self.sy[w][i]) for i in xrange(self.meta.ntimes)] for w in xrange(self.meta.nwalkers)])#/float(self.meta.nwalkers*self.meta.ntimes*self.meta.nbins)
#         var_s = np.average([[np.dot(self.sey[w][i],self.sey[w][i]) for i in xrange(self.meta.ntimes)] for w in xrange(self.meta.nwalkers)])#/float(self.meta.nwalkers*self.meta.ntimes*self.meta.nbins)
#         self.var_ls.append(var_ls)
#         self.var_s.append(var_s)
        #self.tot_var_ls = self.tot_var_ls+var_ls
        #self.tot_var_s = self.tot_var_s+var_s
        #print(self.meta.name+' var_ls='+str(self.var_ls))
        #print(self.meta.name+' var_s='+str(self.var_s))
        return(var)

    def calcchi(self,var,s,data):
        """chi^2 (or Wald test) of samples"""
        v = np.sum([np.average([statistics.variance(walk) for walk in data.T[b]]) for b in xrange(len(data.T))])#abs(np.linalg.det(np.cov(flatdata)))
        ans = np.average(s**2)/v
        var.append(ans)

#         flatdata = np.array([self.ydata.T[b].flatten() for b in xrange(self.meta.nbins)])
#         eflatdata = np.exp(flatdata)

#         vy = np.sum([np.average([statistics.variance(walk) for walk in self.ydata.T[b]]) for b in xrange(self.meta.nbins)])#abs(np.linalg.det(np.cov(flatdata)))
#         vey = np.sum([np.average([statistics.variance(walk) for walk in self.eydata.T[b]]) for b in xrange(self.meta.nbins)])#abs(np.linalg.det(np.cov(eflatdata)))

#         chi_ls = np.average(self.sy**2)/vy#np.average(sp.stats.chisquare(flatdata.T)[0])#float(self.meta.nwalkers*self.meta.ntimes*self.meta.nbins*vy)
#         chi_s = np.average(self.sey**2)/vey#np.average(sp.stats.chisquare(eflatdata.T)[0])#float(self.meta.nwalkers*self.meta.ntimes*self.meta.nbins*vey)
#         self.chi_ls.append(chi_ls)
#         self.chi_s.append(chi_s)
#         #self.tot_chi_ls = self.tot_chi_ls+chi_ls
#         #self.tot_chi_s = self.tot_chi_s+chi_s
#         print(self.meta.name+' chi_ls='+str(self.chi_ls))
#         print(self.meta.name+' chi_s='+str(self.chi_s))
        return(var)

class stat_probs(calcstats):
    """
    calculates statistics requiring only probabilities:  log posterior probability for alternatives, variance of probabilities
    """
    def __init__(self, meta):
        calcstats.__init__(self, meta)
        #self.summary = 0
        self.name = 'probs'

#         # calculating log likelihood ratio test statistic for each relative to truth (and true relative to prior)
        if self.meta.logtruNz is not None:
            self.lp_tru = self.meta.postdist.lnprob(self.meta.logtruNz)
            self.lik_tru = self.meta.postdist.lnlike(self.meta.logtruNz)
        else:
            self.lp_tru = self.meta.postdist.lnprob(self.meta.mean)
            self.lik_tru = self.meta.postdist.lnlike(self.meta.mean)

        self.lp_int = self.meta.postdist.lnprob(self.meta.logintNz)
        self.lp_stk = self.meta.postdist.lnprob(self.meta.logstkNz)
        self.lp_map = self.meta.postdist.lnprob(self.meta.logmapNz)
        self.lp_exp = self.meta.postdist.lnprob(self.meta.logexpNz)
        self.lp_mml = self.meta.postdist.lnprob(self.meta.logmmlNz)

        self.var_y = []

        outdict = {'var_y': self.var_y,
                   'lp_tru': self.lp_tru,
                   'lp_int': self.lp_int,
                   'lp_stk': self.lp_stk,
                   'lp_map': self.lp_map,
                   'lp_exp': self.lp_exp,
                   'lp_mml': self.lp_mml
                  }

        if self.meta.plotonly == False:
            with open(os.path.join(self.meta.topdir,'stat_probs.p'),'wb') as statprobs:
                cpkl.dump(outdict,statprobs)

    def compute(self, ydata):
        y = np.swapaxes(ydata,0,1).T
        var_y = sum([statistics.variance(y[w]) for w in xrange(self.meta.nwalkers)])/self.meta.nwalkers
        #self.llr_smp.append((2.*np.max(lik_y)-2.*self.ll_tru))
        self.var_y.append(var_y)
        # self.summary = self.summary+var_y

        with open(os.path.join(self.meta.topdir,'stat_probs.p'),'rb') as indict:
            outdict = cpkl.load(indict)

        outdict['var_y'] = self.var_y

        with open(os.path.join(self.meta.topdir,'stat_probs.p'),'wb') as statprobs:
            cpkl.dump(outdict,statprobs)

#         return { #'summary': self.summary,
#                  'var_y': self.var_y,
#                  'lp_tru': self.lp_tru,
# #                  'lp_stk': self.lp_stk,
# #                  'lp_map': self.lp_map,
# # #                  'lp_exp': self.lp_exp
#                  'lp_mml': self.lp_mml
#                }

# class stat_fracs(calcstats):
#     """
#     calculates summary statistics on acceptance fractions
#     """
#     def __init__(self, meta):
#         calcstats.__init__(self, meta)
#         self.var_y = []
#         self.name = 'fracs'
#     def compute(self, ydata):
#         y = ydata.T
#         var_y = statistics.variance(y)
#         self.var_y.append(var_y)
#         return {'var_y': self.var_y}

class stat_times(calcstats):
    """
    calculates summary statistics on autocorrelation times
    """
    def __init__(self, meta):
        calcstats.__init__(self, meta)
        self.var_y = []
        self.name = 'times'
    def compute(self, ydata):
        y = ydata.T
        var_y = np.var(y)
        self.var_y.append(var_y)
        return {'var_y': self.var_y}

def cft(xtimes,lag):
    """calculate autocorrelation times since emcee sometimes fails"""
    lent = len(xtimes)-lag
    allt = xrange(lent)
    ans = np.array([xtimes[t+lag]*xtimes[t] for t in allt])
    return ans

def cf(xtimes):#xtimes has ntimes elements
    cf0 = np.dot(xtimes,xtimes)
    allt = xrange(len(xtimes)/2)
    cf = np.array([sum(cft(xtimes,lag)[len(xtimes)/2:]) for lag in allt])/cf0
    return cf

def cfs(x,mode):#xbinstimes has nbins by ntimes elements
    if mode == 'walkers':
        xbinstimes = x
        cfs = np.array([sum(cf(xtimes)) for xtimes in xbinstimes])/len(xbinstimes)
    if mode == 'bins':
        xwalkerstimes = x
        cfs = np.array([sum(cf(xtimes)) for xtimes in xwalkerstimes])/len(xwalkerstimes)
    return cfs

def acors(xtimeswalkersbins,mode):
    if mode == 'walkers':
        xwalkersbinstimes = np.swapaxes(xtimeswalkersbins,1,2)#nwalkers by nbins by nsteps
        taus = np.array([1. + 2.*sum(cfs(xbinstimes,mode)) for xbinstimes in xwalkersbinstimes])#/len(xwalkersbinstimes)# 1+2*sum(...)
    if mode == 'bins':
        xbinswalkerstimes = xtimeswalkersbins.T#nbins by nwalkers by nsteps
        taus = np.array([1. + 2.*sum(cfs(xwalkerstimes,mode)) for xwalkerstimes in xbinswalkerstimes])#/len(xwalkersbinstimes)# 1+2*sum(...)
    return taus

def calckl(difs,lqn,lpn):
    """KL Divergence test"""
    pn = np.exp(lpn)*difs
    qn = np.exp(lqn)*difs
    p = pn/np.sum(pn)
    q = qn/np.sum(qn)
    logp = um.safelog(p)
    logq = um.safelog(q)
    klpq = np.sum(p*(logp-logq))
    klqp = np.sum(q*(logq-logp))
    return(klpq,klqp)

def calcbfe(samples):
    with open(samples,'rb') as csvfile:
        tuples = (line.split(None) for line in csvfile)
        alldata = [[float(pair[k]) for k in range(0,len(pair))] for pair in tuples][1:]
        nbins = len(alldata[0])
        alldata = np.array(alldata).T

    locs,scales = [],[]
    for k in xrange(nbins):
        y_all = alldata[k].flatten()
        loc,scale = sp.stats.norm.fit_loc_scale(y_all)
        locs.append(loc)
        scales.append(scale)
    locs = np.array(locs)
    scales = np.array(scales)
    return(locs,scales)

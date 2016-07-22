"""
key-mcmc module defines loading/saving/naming conventions for MCMC
"""

# TO DO: redo state handling with dicts or generators rather than lists

from utilmcmc import path
import distribute
import cPickle
import hickle as hkl
import os

def safe_load_c(path, num_objs = None):
    """read cPickle"""
    print 'loading: ' + path
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        if num_objs is None:
            return cPickle.load(f)
        return [cPickle.load(f) for _ in xrange(num_objs)]
    f.close()
def safe_store_c(path, o):
    """write cPickle"""
    print 'storing: ' + path
    directory = path[:path.rfind('/')]
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(path, "wb") as f:
        cPickle.dump(o, f)
    f.close()

def safe_load_h(path, num_objs = None):
    """read hickle"""
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        if num_objs is None:
            return hkl.load(f)
        return [hkl.load(f) for _ in xrange(num_objs)]
    f.close()
def safe_store_h(path, o):
    """write hickle"""
#     print 'storing hkl:' + path
    directory = path[:path.rfind('/')]
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(path, "w") as f:
        hkl.dump(o, f)
    f.close()

# define templates for path
state_builder = path("{topdir}/state-{r}.p")#hkl")
iterno_builder = path("{topdir}/iterno.p")
statistics_builder = path("{topdir}/stat_{stat_name}.p")

class key(distribute.distribute_key):
    """all the dict handling here"""
    def __init__(self, **kv):
        self.t = None
        self.r = None
        self.burnin = None
        self.update_dict(kv)

    def copy(self):
        ret = key()
        ret.t = self.t
        ret.r = self.r
        ret.burnin = self.burnin
        return ret

    def __hash__(self):
        return hash((self.t, self.r, self.burnin))

    def __eq__(self, other):
        if not isinstance(other, key):
            return False
        return (self.t == other.t and
                self.r == other.r and
                self.burnin == other.burnin)

    def __repr__(self):
        return ("KEY: " + str(self.to_dict()))

    def update_dict(self, d):
        if 't' in d:
            self.t = d['t']
        if 'r' in d:
            self.r = d['r']
        if 'burnin' in d:
            self.burnin = d['burnin']

    def add_dict(self, d):
        ret = self.copy()
        ret.update_dict(d)
        return ret

    def add(self, **d):
        ret = self.copy()
        ret.update_dict(d)
        return ret

    def filter(self, s):
        newKey = key()
        if 't' in s:
            newKey.t = self.t
        if 'r' in s:
            newKey.r = self.r
        if 'burnin' in s:
            newKey.burnin = self.r
        return newKey

    def to_dict(self, d = None):
        retval = d
        if retval is None:
            retval = {}
        if self.t is not None:
            retval['t'] = self.t
        if self.r is not None:
            retval['r'] = self.r
        if self.burnin is not None:
            retval['burnin'] = self.burnin
        return retval

    # should break this up around here to use inheritance rather than lumping everything into one class

    def load_state(self, topdir):
        """state is mutable permcmc object at each stage"""
        filepath = state_builder.construct(**self.to_dict({'topdir':topdir}))
        return safe_load_c(filepath)#safe_load_h(filepath)
    def store_state(self, topdir, o):
        """state is mutable permcmc object at each stage"""
        filepath = state_builder.construct(**self.to_dict({'topdir':topdir}))
        safe_store_c(filepath,o)#safe_store_h(filepath, o)

    def load_iterno(self, topdir):
        """need to know iterno for progress"""
        filepath = iterno_builder.construct(**self.to_dict({'topdir':topdir}))
        return safe_load_c(filepath)
    def store_iterno(self, topdir, o):
        """need to know iterno for progress"""
        filepath = iterno_builder.construct(**self.to_dict({'topdir':topdir}))
        safe_store_c(filepath, o)

    def load_stats(self, topdir, name, size):
        """intermediate stats now stored as a series of pickled entries, rather than one large list, need to know length of list to read it"""
        filepath = statistics_builder.construct(**self.to_dict({'topdir':topdir, 'stat_name':name}))
        return safe_load_c(filepath, size)
    def add_stats(self, topdir, name, o):
        """intermediate stats now stored as a series of pickled entries, rather than one large list"""
        filepath = statistics_builder.construct(**self.to_dict({'topdir':topdir, 'stat_name':name}))
        with open(filepath, "ab") as f:
            cPickle.dump(o, f)

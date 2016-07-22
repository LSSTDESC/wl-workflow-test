"""
distribute module sets up for multiprocessing with communication between workers and consumers
"""

import multiprocessing as mp
import sys
import traceback

class distribute_key(object):
    """class to access data that's been serialized to disk"""
    def __init__(self):
        pass

class consumer(object):
    """consumer is the plotter, producer is MCMC calculation"""
    def __init__(self, *args):
        pass
    def loop(self, queue):
        while (True):
            key = queue.get()
            print('getting key '+str(key))
            if (key=='done'):
                self.finish()
                return
            self.handle(key)
    def finish(self):
        print "BAD: Generic finish called."

def run_offthread(func, *args):
   proc = mp.Process(target=func, args=args)
   proc.start()
   return proc
def run_offthread_sync(func, *args):
   run_offthread(func, *args).join()

# def run_offthread(pool, func, *args):
#     proc = pool.apply_async(func, args)
#     return proc
# def run_offthread_sync(pool, func, *args):
#     print 'func: ' + str(func)
#     return pool.apply(func, args)


def do_consume(ctor, q, args):
    """plot results of one run in loop"""
    #print(ctor,q,args)
    try:
        obj = ctor(**args)
        obj.loop(q)
    except:
        e = sys.exc_info()[0]
        print 'Nested Exception: '
        print traceback.format_exc()
        sys.stdout.flush()
        raise e
def run_consumer(ctor, q, args):
    return run_offthread(do_consume, ctor, q, args)

class distribute(object):
    """
    class distributes computation over multiple threads
    consumers is list of consumers called and added to queue whenever a new key is complete
    """
    def __init__(self, consumers, start = True, **args):
        self.queues = [mp.Queue() for _ in consumers]
#         for q in self.queues:
#             key = q.get()
#             print('original key '+str(key))

        self.consumer_lambdas = consumers
        self.consumers = [run_consumer(c,q, args) for (c,q) in zip(self.consumer_lambdas, self.queues)]
        self.started = False

    def complete_chunk(self, key):
        """called by producer when chunk of data has been produced"""
        for q in self.queues:
            q.put(key)
            print('putting key '+str(key))

    def finish(self):
        """called when all producers are done"""
        for q in self.queues:
            q.put('done')
#             print('put key '+str('done'))

    def finish_and_wait(self):
        self.finish()
        for c in self.consumers:
            c.join()

    def run(self, args):
        """start the consumers"""
        if self.started:
            return
        self.started = True
        print('Starting {} threads'.format(len(self.consumers)))
        for t in self.consumers:
            print ('starting: {}'.format(str(t)))
            t.start()

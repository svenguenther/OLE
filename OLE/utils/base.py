"""
Here we are going to define a base class. It will be the parent class of all other classes. 

It has a timer function that will be used by all other classes.

It has a logger function that will be used by all other classes.

The buildup is inspired by Cobaya (https://github.com/CobayaSampler) by Jesus Torrado and Antony Lewis.

"""

import time
import logging
from .mpi import *

c_km_s = 299792.458

class SubTimer:
    def __init__(self, name = None):
        self._name = name
        self._start = None
        self._first_time = None
        self.n = 0
        self.time_sum = 0

    def start(self):
        self._start = time.time()
        if self._first_time is None:
            self._first_time = self._start

    def time_from_start(self):
        return time.time() - self._start
    
    def time_from_first(self):
        return time.time() - self._first_time
    
    def get_time_avg(self):
        return self.time_sum / self.n
    
    def increment(self):
        self.n += 1
        self.time_sum += self.time_from_start()


class Timer:
    def __init__(self, name = None):
        self._start_time = time.time()  
        self.subtimer = {}
        self.last_round = 0.0

    def _init_sub_timer(self, name):
        self.subtimer[name] = SubTimer(name)

    def start(self, name):
        if name not in self.subtimer:
            self._init_sub_timer(name)
        self.subtimer[name]._start = time.time()

    def time_from_start(self, name):
        return time.time() - self.subtimer[name]._start
    
    def time_from_init(self):
        return time.time() - self._start_time

    def n(self, name):
        return self.subtimer[name].n
        
    def get_time_avg(self, name):
        return self.subtimer[name].get_time_avg()
    
    def get_summed_time(self, name):
        return self.subtimer[name].time_sum

    def increment(self, name):
        self.subtimer[name].n += 1
        self.subtimer[name].time_sum += self.time_from_start(name)
        self.subtimer[name].last_round = self.time_from_start(name)

    def log_time(self, name, logger):
        # do logging but give only 3 decimals
        logger.info(f"Timing for {name}: total: {self.get_summed_time(name):.4f}s calls: {self.n(name)} avg: {self.get_time_avg(name):.4f}s")

    def log_all_times(self, logger):
        for name in self.subtimer:
            self.log_time(name, logger)

class Logger:
    def __init__(self, name = None):
        self._name = name or self.__class__.__name__
        self.logger = logging.getLogger(self._name)

        # set the rank of the logger
        if get_mpi_size() > 1:
            self.rank = str(get_mpi_rank())
        else:
            self.rank = str(0)

        # make nice logging format
        if get_mpi_size() > 1:
            formatter = logging.Formatter('%(asctime)s - %(name)s - rank: ' + str(self.rank) +  ' - %(levelname)s - %(message)s')
        else:
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

    def rename(self, name):
        self._name = name
        self.logger.name = name

    def set_logger(self, logger):
        self.logger = logger

    def get_logger(self):
        return self.logger

    def set_loglevel(self, level):
        self.logger.setLevel(level)

    def get_loglevel(self):
        return self.logger.getEffectiveLevel()

    def log(self, level, msg, *args, **kwargs):
        self.logger.log(level, msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

class BaseClass(Logger, Timer):
    debug_mode: bool
    initialized: bool
    def __init__(self, name=None, debug= False, **kwargs):
        # init both parent classes
        super().__init__(name)
        super(Logger, self).__init__(name)
        self.set_loglevel(logging.DEBUG if debug else logging.INFO)
        self.debug_mode = debug
        self.initialized = False
    
    def initialize(self, **kwargs):
        self.initialized = True
        if 'debug' in kwargs:
            self.debug_mode = kwargs['debug']
            self.set_loglevel(logging.DEBUG if self.debug_mode else logging.INFO)
            
        pass



class constant:
    def __init__(self, value=1):
        self.value = value

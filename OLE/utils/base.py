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

class Timer:
    def __init__(self, name = None):
        self.n = 0
        self.time_sum = 0.
        self._start = None
        self._first_time = None

    def start(self):
        self._start = time.time()

    def time_from_start(self):
        return time.time() - self._start

    def n_avg(self):
        return self.n - 1 if self.n > 1 else self.n

    def get_time_avg(self):
        if self.n > 1:
            return self.time_sum / (self.n - 1)
        else:
            return self._first_time

    def increment(self, logger=None):
        delta_time = time.time() - self._start
        if self._first_time is None:
            if not delta_time:
                logger.warning("Timing returning zero, may be inaccurate")
            # first may differ due to caching, discard
            self._first_time = delta_time
            self.n = 1
            if logger:
                logger.debug("First evaluation time: %g s", delta_time)

        else:
            self.n += 1
            self.time_sum += delta_time
        if logger:
            logger.debug("Total evaluation time: %f s", self.time_sum)
            logger.debug("Number of calls: %d", self.n)
            logger.debug("Average evaluation time: %g s", self.get_time_avg())


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
        self.start()
    
    def initialize(self, **kwargs):
        self.initialized = True
        if 'debug' in kwargs:
            self.debug_mode = kwargs['debug']
            self.set_loglevel(logging.DEBUG if self.debug_mode else logging.INFO)
            
        pass



class constant:
    def __init__(self, value=1):
        self.value = value

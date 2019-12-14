import time
from functools import wraps
from base_log import get_logger

# Decolator Method
def stop_watch(EXP_VERSION):

    def _stop_watch(func):
        @wraps(func)
        def wrapper(*args, **kargs):
            start_time = time.time()

            func(*args, **kargs)

            elapsed_time = int(time.time() - start_time)
            minits, sec = divmod(elapsed_time, 60)
            hour, minits = divmod(minits, 60)

            get_logger(EXP_VERSION).info("[elapsed_time]\t>> {:0>2}:{:0>2}:{:0>2}".format(hour, minits, sec))
        return wrapper

    return _stop_watch
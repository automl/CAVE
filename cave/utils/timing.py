from functools import wraps
from time import time
import logging


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        logger = logging.getLogger("cave.timer")
        ts = time()
        result = f(*args, **kw)
        te = time()
        seconds = te-ts
        h_m_s = (seconds // 60**2, (seconds // 60) % 60, seconds % 60)
        logger.debug('func:%r took: %2.4f sec (human-friendly: %d h /%d m /%d s)', f.__name__, te-ts, *h_m_s)
        return result
    return wrap

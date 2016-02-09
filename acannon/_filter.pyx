import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

def running_median_trend(np.ndarray[DTYPE_t, ndim=1, mode='c'] x,
                         np.ndarray[DTYPE_t, ndim=1, mode='c'] y,
                         int hw=500):
    cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] r = np.empty(len(y), dtype=DTYPE)
    cdef int i, a, b
    for i in range(x.shape[0]):
        a = i - hw
        if a < 0:
            a = 0
        b = i + hw
        if b > x.shape[0]:
            b = x.shape[0]
        r[i] = np.median(y[a:b])
    return r

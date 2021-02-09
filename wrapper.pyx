import numpy as np
cimport numpy as np

assert sizeof(int) == sizeof(np.int32_t)

assert sizeof(float) == sizeof(np.float32_t)

cdef extern from "src/kmeans.h":
    cdef cppclass C_kmeans "kmeans":
        void getClusters(np.int32_t*, np.float32_t*, int, int, int, int)

cdef class kmeans:
    cdef C_kmeans* km
    def getClusters(self, np.ndarray[ndim=1, dtype=np.float32_t] arr, n, dim, k, iter):
        cdef np.ndarray[ndim=1, dtype=np.int32_t] a = np.zeros(n, dtype=np.int32)
        self.km.getClusters(&a[0], &arr[0], n, dim, k, iter)
        
        return a
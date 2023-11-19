#######################################
## author: Jakob Buchheim
## company: Columbia University
##
## version history:
## 		* created: 20190914
##
## description:
##  * sets environment variable to supress unwanted multithreading
##  * copy from https://github.com/numpy/numpy/issues/11826#issuecomment-476954711
#######################################

import os, ctypes
import numexpr


def setENV_threads(numThreads):
    """
    Very surprising if users should use this function directly -- this propagates though various mkl explicit thread changes to deal with the way ENV variables are set on the linracks.

    Parameters
    ----------
    numThreads : int
        The desired thread count we're using, typically set globally in the makefile and then passed in as an arg via argparse.

    Returns
    -------
    None.
    """
    try:
        import mkl

        mkl.set_num_threads(numThreads)
        return 0
    except:
        pass

    for name in ["libmkl_rt.so", "libmkl_rt.dylib", "mkl_Rt.dll"]:
        try:
            mkl_rt = ctypes.CDLL(name)
            mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(1)))
            return 0
        except:
            pass

    os.environ["OMP_NUM_THREADS"] = str(numThreads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(numThreads)
    os.environ["MKL_NUM_THREADS"] = str(numThreads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(numThreads)
    # os.environ["NUMEXPR_NUM_THREADS"] = str(numThreads)
    # os.environ['NUMEXPR_MAX_THREADS'] = str(numThreads+1)
    os.environ["NUMBA_NUM_THREADS"] = str(numThreads)
    os.environ["MKL_DYNAMIC"] = "FALSE"

    # numexpr.set_num_threads(numThreads)
    # numexpr.set_vml_num_threads(numThreads)
    return

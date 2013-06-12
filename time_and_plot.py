
from matplotlib import ticker
import numpy as np
from timeit import timeit
from matplotlib import pyplot as plt

def time_and_plot(algo, Ns, dtype, func_names, time_func, xlabel, ylabel, yscale = 'linear', yscaling = 1):
    """Plot Timing Comparison

    Timing Parameters
    -----------------
    algo : str
        Name of the algorithm being timed.  Example: "Matrix-Matrix Multiply",
    Ns : indexable, int
        Size of arguments to be passed to timed functions.
    dtype : numpy.dtype
        Type of arguments to pass to timed functions.
    func_names : str
        Name of the functions in the __main__ namespace to be timed.
    time_func : function
        Timing function.  See time_func.

    Plot Parameters
    ---------------
    xlabel : str
        See plt.xlabel.
    ylabel  : str
        See plt.ylabel.
    yscale  : str, optional, defaults to 'linear'
        See plt.set_yscale.
    yscaling : integer, optional, defaults to 1
        Ratio to multiply y data values by.
    """

    data = np.empty((len(func_names), len(Ns)), dtype=np.float)
    for k in xrange(len(func_names)):
        for i in xrange(len(Ns)):
            data[k,i] = time_func(func_names[k], Ns[i], dtype)

    plt.clf()
    fig1, ax1 = plt.subplots()
    w, h = fig1.get_size_inches()
    fig1.set_size_inches(w*1.5, h)
    ax1.set_xscale('log')
    ax1.get_xaxis().set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax1.get_xaxis().set_minor_locator(ticker.NullLocator())
#    ax1.set_xticks(Ns)
    ax1.set_yscale(yscale)
    plt.setp(ax1.get_xticklabels(), fontsize=14)
    plt.setp(ax1.get_yticklabels(), fontsize=14)

    ax1.grid(color="lightgrey", linestyle="--", linewidth=1, alpha=0.5)

    plt.xlabel(xlabel, fontsize=24)
    plt.ylabel(ylabel, fontsize=24)
    plt.xlim(Ns[0]*.9, Ns[-1]*1.1)
    plt.suptitle("%s Performance" % (algo), fontsize=24)

    for k in xrange(len(func_names)):
        plt.plot(Ns, data[k,:]*yscaling, 'o-', linewidth=2, markersize=5, label=func_names[k])
        plt.legend(loc='upper right', fontsize=18)

def time_sum_func(func_name, N=10000, dtype=np.int32, trials=100):
    """Timeit Helper Function (Simple One-D Functions)

    Parameters
    ----------
    func_name : str
        Name of the function in the __main__ namespace to be timed.
    N : int, optional, defaults to 10000
        Size of np.ones array to construct and pass to function.
    dtype : np.dtype
        Type of array to construct and pass to function.
    trials : int, optional, defaults to 100
        This parameter is passed to timeit

    Returns
    -------
    func_time : float
        Average execution time over all trials
    """
    import __main__
    __main__.y = np.ones((N), dtype=dtype)
    return (timeit(stmt="func(__main__.y)",
                   setup="import __main__; from __main__ import %s as func" % func_name,
                   number=trials)/trials)


def time_kernel(k_name, N, dtype):
    """Timeit Helper Function (GrowCut Functions)

    Parameters
    ----------
    k_name : str
        Name of the GrowCut kernel in the __main__ namespace to be timed.
    N : int
        Size of image arrays to construct and pass to function.
    dtype : np.dtype
        Type of arrays to construct and pass to function.

    Returns
    -------
    func_time : float
        Average execution time over 3 trials
    """

    import __main__
    image = np.zeros((N, N, 3), dtype=dtype)
    state = np.zeros((N, N, 2), dtype=dtype)
    state_next = np.empty_like(state)

    # colony 1 is strength 1 at position 0,0
    # colony 0 is strength 0 at all other positions
    state[0, 0, 0] = 1
    state[0, 0, 1] = 1

    __main__.image = image
    __main__.state = state
    __main__.state_next = state_next

    trials = 3

    return timeit(stmt="kernel(__main__.image, __main__.state, __main__.state_next, 10)",
                  setup="from __main__ import %s as kernel; import __main__" % k_name,
                  number=trials)/trials

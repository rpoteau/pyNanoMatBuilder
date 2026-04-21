# Try to import Numba for Just-In-Time (JIT) compilation and parallel loops
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Fallback: Define a dummy njit decorator if Numba is not installed
    def njit(*args, **kwargs):
        def decorator(f):
            return f
        return decorator
    # Fallback: map prange to standard range for sequential execution
    prange = range

# You can now use HAS_NUMBA anywhere to check the status
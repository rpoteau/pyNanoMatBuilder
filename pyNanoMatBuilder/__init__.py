"""
pyNanoMatBuilder
A versatile Python library designed to generate atomic-scale 3D structures of nanoparticles (NPs)
"""

__version__ = "0.11.4"
__last_update__ = "2026-05-03"
__author__ = "Sara Mokhtari, Romuald Poteau"

# --- Numba Protection & Parallelism Configuration ---
import os
_HAS_NUMBA = False
_DEFAULT_THREADS = 1

try:
    import numba
    _HAS_NUMBA = True
    
    # Internal function to configure Numba threading
    def _setup_parallelism(n_threads=None):
        total_cores = os.cpu_count() or 1
        if n_threads is None:
            # Default policy: half of the cores, at least 1
            n_threads = max(1, total_cores // 2)
        
        try:
            numba.set_num_threads(n_threads)
            return n_threads
        except Exception:
            return 1

    # Initial setup
    _current_threads = _setup_parallelism()

except ImportError:
    # If numba is missing, we define dummy functions to avoid NameErrors
    _current_threads = 1
    
    def _setup_parallelism(n_threads=None):
        return 1

# --- Public API for thread management ---
def set_threads(n):
    """
    Sets the number of threads for parallel calculations.
    Does nothing if numba is not installed.
    """
    global _current_threads
    if _HAS_NUMBA:
        _current_threads = _setup_parallelism(n)
        print(f"pyNanoMatBuilder: parallelism set to {_current_threads} threads.")
    else:
        print("pyNanoMatBuilder: numba not installed. Staying in single-thread mode.")

def get_threads():
    """Returns the number of threads currently used."""
    return _current_threads
    
# Expose data and utils modules
from . import data
from . import utils as pyNMBu

# Fast import: move internal objects to the top-level namespace
# This allows: pyNMB.init() instead of pyNMB.visualID.init()
from . import visualID as vID

from .visualID import (
    init, end, 
)
from .utils import (
    fg, hl, bg, color, 
    centerTitle, centertxt
)

from .utils.geometry import plot_npr_triangle

# Define which symbols are exported when someone does 'from pyNanoMatBuilder import *'
__all__ = [
    "fg", "hl", "bg", "color", "init", "end",
    "chrono_start", "chrono_stop", "chrono_show",
    "centerTitle", "centertxt", "data", "pyNMBu",
    "plot_npr_triangle"
]

from .pyNMBcore import pyNMBcore
from_file = pyNMBcore.from_file
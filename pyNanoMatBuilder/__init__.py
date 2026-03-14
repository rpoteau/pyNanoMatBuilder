"""
pyNanoMatBuilder
A versatile Python library designed to generate atomic-scale 3D structures of nanoparticles (NPs)
"""

__version__ = "0.8.0"
__last_update__ = "2026-03-15"
__author__ = "Sara Mokhtari, Romuald Poteau"


# Expose data and utils modules
from . import data
from . import utils as pyNMBu

# Fast import: move internal objects to the top-level namespace
# This allows: pyNMB.init() instead of pyNMB.visualID.init()
from . import visualID as vID
from .visualID import (
    fg, hl, bg, color, 
    init, end, 
    centerTitle, centertxt
)

# Define which symbols are exported when someone does 'from pyNanoMatBuilder import *'
__all__ = [
    "fg", "hl", "bg", "color", "init", "end",
    "chrono_start", "chrono_stop", "chrono_show",
    "centerTitle", "centertxt", "data", "pyNMBu"
]

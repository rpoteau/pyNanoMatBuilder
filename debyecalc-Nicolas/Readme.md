This folder contains code for Debye calculations. 
Required packages: debyecalculator, torch


At the present time, 2 functions are implemented:
- calcSofQ (structure file, qmin, qmax,qstep,biso)
- calcGofR (structure file, rmin,rmax,rstep,biso,qdamp)

  Those 2 functions can be inserted in utils.py for further use in pyNanoMatBuilder library, or as a (common) method to all classes from the library. There might be more elegant ways.

For memory, a code is provided to perform refinement of a structural model against experimental PDF or S(q). This code is based on lmfit python library (file FitPDF.py)

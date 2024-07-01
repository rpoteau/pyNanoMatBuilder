This folder contains code for Debye calculations.
Debye calculations are based on debyecalculator package. It allows computations of scattering curves S(q) and pair distribution functions G(r) from xyz files generated using pyNanoMatBuilder library.

At the present time, 2 functions are implemented:
- calcSofQ (structure file, qmin, qmax,qstep,biso)
- calcGofR (structure file, rmin,rmax,rstep,biso,qdamp)

  Those 2 functions can be inserted in utils.py for further use in pyNanoMatBuilder library.

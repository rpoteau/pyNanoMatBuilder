This folder contains source codes for Debye calculations. Here 7 solutions were identified, they are presented here for memories. 
These solutions accept different file formats and use different expressions for atomic scattering factor (sometimes neglecting the dispersive component).
The following solutions can be used:
- debyer code (external program https://debyer.readthedocs.io/en/latest/) takes xyz file as input - only the non dispersive component is taken into account (no anomalous effect) FILE: no file provided
  
- crysol (external program https://www.embl-hamburg.de/biosaxs/manuals/crysol.html#cli) - takes pdb format as input - can take into account the dispersive component for anomalous investigations FILE:pentarod_crysol.py
  
- D+ (python library - windows only https://dplus-python-api.readthedocs.io/en/latest/) - takes pdb format as input - can handle large and hierarchical structures and is GPU friendly. No anomalous investigations. FILE: pentarod_Marianne_Dplus.ipynb
  
- sasview (python library) - takes pdb as input - does not account for anomalous effects. (https://www.sasview.org/docs/user/qtgui/Calculators/sas_calculator_help.html#) FILE:Iofq_nodiffpy_py311
  
- 'diffpy': it is a python function, named Iofq, that uses diffpy-cmi library to load the structure, and get statistics from it (such as pairs). This function is open and is a good candidate to adapt to pyNanoMatBuilder. Scaterring factors can be computed as desired, e.g. using xraylarch library FILE: Iofq_diffpy
  
- Python function derived from diffpy function described above, but modified to take xyz files as input : complex scattering factors, talking into account anomalous effects, are computed using xraylarch library.  FILE: Iofq_fcomplex_ok, is a very good candidate for optimisation. However there is some work to be done (introduce fNT is structure factor)
  
- DebyeCalculator python package: provides same results as Debyer and can compute PDF (in fact same as Debyer but within Python). Is optimized for GPU calculations. No anomalous calculations. See file pyNMB_Debyecalc for (straightforard) implementation with pyNMB. FitPDF.py is a file that can be used as starting point for PDF or I(q) refinemant algorithm.




  For clarity, here is the full expression of X-ray form factor in its complex form, which appears in Debye equation:

  f(q,energy)=f_0(q)+f'(energy,Z)+f_NT+i*f''(energy)   (here f0, f' and f'' are provided by xraylarch)
  f(q,energy)=f_0(q)+f_1(energy)+frel-Z+f_NT+i*f_2(energy) in which all terms (except f0) can be extracted from NIST database




This folder contains source codes for Debye calculations. Here 6 solutions were identified, they are presented here for memories. 
These solutions accept different file formats and use different expressions for atomic scattering factor (sometimes neglecting the dispersive component).
The following solutions can be used:
- debyer code (external program https://debyer.readthedocs.io/en/latest/) takes xyz file as input - only the non dispersive component is taken into account (no anomalous effect) FILE:
- crysol (external program https://www.embl-hamburg.de/biosaxs/manuals/crysol.html#cli) - takes pdb format as input - can take into account the dispersive component for anomalous investigations FILE:pentarod_crysol.py
- D+ (python library - windows only https://dplus-python-api.readthedocs.io/en/latest/) - takes pdb format as input - can handle large and hierarchical structures and is GPU friendly. No anomalous investigations. FILE: pentarod_Marianne_Dplus.ipynb
- sasview (python library) - takes pdb as input - does not account for anomalous effects. (https://www.sasview.org/docs/user/qtgui/Calculators/sas_calculator_help.html#) FILE:Iofq_nodiffpy_py311
- 'diffpy': it is a python function, named Iofq, that uses diffpy-cmi library to load the structure, and get statistics from it (such as pairs). This function is open and is a good candidate to adapt to pyNanoMatBuilder. Scaterring factors can be computed as desired, e.g. using xraylarch library FILE: Iofq_diffpy
- Python function derived from diffpy function described above, but modified to take xyz files as input : complex scattering factors, talking into account anomalous effects, are computed using xraylarch library.  FILE: Iofq_fcomplex_ok, is a very good candidate for optimisation


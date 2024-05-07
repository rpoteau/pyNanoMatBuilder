This folder contains source codes for Debye calculations. Here 6 solutions were identified, they are presented here for memories. 
These solutions accept different file formats and use different expressions for atomic scattering factor (sometimes neglecting the dispersive component).
The following solutions can be used:
- debyer code (external program) takes xyz file as input - only the non dispersive component is taken into account (no anomalous effect) FILE:
- crysol (external program) - takes pdb format as input - can take into account the dispersive component for anomalous investigations FILE:
- D+ (python library - windows only) - takes pdb format as input - can handle large and hierarchical structures and is GPU friendly. No anomalous investigations. FILE: pentarod_Marianne_Dplus.ipynb
- sasview (python library) - takes pdb as input - does not account for anomalous effects.
- 'diffpy': it is a python function, named Iofq, that uses diffpy-cmi library to load the structure, and get statistics from it (such as pairs). This function is open and is a good candidate to adapt to pyNanoMatBuilder. Scaterring factors can be computed as desired, e.g. using xraylarch library FILE: Iofq_diffpy
- Python function derived from A. Boule notebook (see https://github.com/aboulle/HPPython/blob/master/Debye.ipynb) : FILE: Iofq_nodiffpy_py311 also a good candidate


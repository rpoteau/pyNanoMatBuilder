import os
import numpy as np
from matplotlib import pyplot as plt

path='/home-local/ratel-ra/Documents/SAXS_data/Data_pentarods/'
pdbfile=path+'structures/Decahedron_6_1_0.pdb'

datafile=path+'Decahedron_6_1_0_D_plus.iq'
output_crysol=path+'Decahedron_6_1_0'

os.system("crysol %s"%pdbfile+" --units 1 --ns 10000 --smax 2 -p %s"%str(output_crysol))

calcfile=output_crysol+'.abs'



qexp,Iexp=np.loadtxt(datafile)
qcalc,Icalc=np.loadtxt(calcfile,unpack=True,skiprows=1)
plt.figure(1)

plt.loglog(0.1*qexp,Iexp,'-k',label='D+')
plt.loglog(qcalc,Icalc,'--r',label='crysol')
plt.xlabel('q (1/angströms)')
plt.ylabel('Normalized intensity')
plt.legend()
plt.show()
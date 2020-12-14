import numpy as np
import matplotlib.pyplot as plt
from ML_functions import gen_events
from density_funcs import rho_baryon

from astropy.io import fits
hdu = fits.open('galaxy1.fits')
data = hdu[1].data

import sys
n = int(sys.argv[1])

def IMF(m):
    #use a Kroupa IMF

    #calculated in a seperate jupyter notebook
    if (m<0.08):
        alpha = 0.3
    else:
        if (m<0.5):
            alpha = 1.3
        else:
            alpha = 2.3


    return np.power(m, -1*alpha)


M = np.logspace(-2, 3, 60)
step = np.roll(M, -1) - M
step[-1] = step[-2]

events = np.zeros([n,30])

norm = 0

for (m,s) in zip(M,step):
    i = IMF(m)*s
    norm = norm+i


print('finished normalizing: ', norm)   

for (m,s) in zip(M,step):
    E, T = gen_events(m, n, 1.0, rho_baryon, data, iso = False, baryons = False)
    E = E[1]

    E = E*IMF(m)*s/norm

   #plt.plot(T, np.sum(E, axis=0))
#    
    print('finished for mass: ', m)
    print(np.sum(E, axis=0))
#    
    events = np.add(events,E)

np.savetxt('baryon_events_LoS_{}.txt'.format(str(n)), events)

    


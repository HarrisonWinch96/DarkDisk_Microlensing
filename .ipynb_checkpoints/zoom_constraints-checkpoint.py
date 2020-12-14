from ML_functions import max_f, gen_events
from density_funcs import rho_NFW, rho_mirrordisk

from astropy.io import fits
hdu = fits.open('galaxy1.fits')
data = hdu[1].data

import numpy as np

M = np.logspace(-2, 1, 30)
F = np.zeros_like(M)
F1 = np.zeros_like(M)

i=0
for m in M:
    F[i] = max_f(m, 100, [1.0,1.0], rho_mirrordisk, err_baryons = 0.05, logfexp = 0, m_b = 0.36, baryons = True, iso = False)
    #F1[i] = max_f(m, 20, [1.0,1.0], rho_mirrordisk, err_baryons = 0.05, logfexp = 0, m_b = 1.0, baryons = True, iso = False)
    print(m, F[i], F1[i])
    i = i+1
np.savetxt('zoom_constraints_full_baryons.txt', F)
#np.savetxt('zoom_constraints_m1.txt', F1)
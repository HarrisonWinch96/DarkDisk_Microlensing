import numpy as np
import matplotlib.pyplot as plt
from ML_functions import gen_events
from density_funcs import rho_baryon

from astropy.io import fits
hdu = fits.open('galaxy1.fits')
data = hdu[1].data

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

events = np.zeros([100,30])

norm = 0

for (m,s) in zip(M,step):
    i = IMF(m)*s
    norm = norm+i
    

print('finished normalizing: ', norm)   

#for (m,s) in zip(M,step):
#    E, T = gen_events(m, 100, 1.0, rho_baryon, data, iso = False, baryons = False)
#    E = E[1]
#    
#    E = E*IMF(m)*s/norm
#    
    #plt.plot(T, np.sum(E, axis=0))
#    
#    print('finished for mass: ', m)
#    print(np.sum(E, axis=0))
#    
#    events = np.add(events,E)
    
#print(events)
#plt.xscale('log')
#plt.yscale('log')

events_core = np.zeros([1,30])
for (m,s) in zip(M,step):
    E, T = gen_events(m, 1, 1.0, rho_baryon, 'core', iso = False, baryons = False)
    E = E[1]
#    
    E = E*IMF(m)*s/norm
#    
    #plt.plot(T, np.sum(E, axis=0))
#    
    print('finished for mass: ', m)
    print(np.sum(E, axis=0))
#    
    events_core = np.add(events_core,E)
    
    
events_LMC = np.zeros([1,30])
for (m,s) in zip(M,step):
    E, T = gen_events(m, 1, 1.0, rho_baryon, 'LMC', iso = False, baryons = False)
    E = E[1]
#    
    E = E*IMF(m)*s/norm
#    
    #plt.plot(T, np.sum(E, axis=0))
#    
    print('finished for mass: ', m)
    print(np.sum(E, axis=0))
#    
    events_LMC = np.add(events_LMC,E)
    
    
#events_one, T = gen_events(0.36, 100, 1.0, rho_baryon, data, iso = False, baryons = False)
#events_one = events_one[1]

#events_iso, T = gen_events(0.36, 100, 1.0, rho_baryon, data, iso = True, baryons = False)
#events_iso = events_iso[1]


#plt.xlabel('crossing time (days)')
#plt.ylabel('number of events per bin')

#plt.plot(T, np.sum(events, axis=0))
#plt.savefig('baryon_event_histogram.png')
#plt.show()

#np.savetxt('baryon_events.txt', events)
np.savetxt('baryon_events_core.txt', events_core)
np.savetxt('baryon_events_LMC.txt', events_LMC)
#np.savetxt('baryon_events_one.txt', events_one)
#np.savetxt('baryon_events_iso.txt', events_iso)
    
    
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import units as u
from astropy import constants as const

import scipy.integrate as integrate

from ML_functions import dGdt_arg

hdu = fits.open('galaxy1.fits')
data = hdu[1].data
data_masked = data[~np.less(np.sqrt(data['glat']**2 + ((data['glon']+180)%360 - 180)**2), 2.0)]
data_masked5 = data[~np.less(np.sqrt(data['glat']**2 + ((data['glon']+180)%360 - 180)**2), 5.0)]

from scipy.stats import norm
from scipy.stats import poisson
from scipy.special import logsumexp

def gen_events(m, N_sources, q, rho_dm, iso=True, data = data, surtime = 4400):
    cutoff = 1.0
    
    N_step = N_sources - 1
    res = np.int(data.size/N_step)
    N_ev = 0
    #i = 1
    
    sources_per_LoS = 17e9/N_sources
    
    llim = np.log10(np.maximum(cutoff,np.sqrt(m)))
    ulim = np.log10(np.maximum(cutoff*100,10000*np.sqrt(m)))
    #print(llim, ulim)
    n = 20
    T = np.logspace(llim, ulim, n)
    N_B = np.zeros((N_sources, n))
    N_D = np.zeros((N_sources, n))
    N_T = np.zeros((N_sources, n))
    
    rat = np.logspace(llim, ulim, n)
    step = T - np.roll(T,1)
    step[0] = 0
    
    i_s = 0
    
    for s in data[::res]:
        
        L = s['rad']*1000*u.parsec
        l = s['glon']
        b = s['glat']
        
        i_t = 0
        for t in T:
            
            N_B[i_s, i_t] = sources_per_LoS*integrate.quad(dGdt_arg, 0.0, 1.0, args = (0.36,L,t,l,b,q,rho_baryon, False))[0]*(t+surtime)*step[i_t] #use average sqrt(mass) insted of average mass
            N_D[i_s, i_t] = sources_per_LoS*integrate.quad(dGdt_arg, 0.0, 1.0, args = (m,L,t,l,b,q,rho_dm, iso))[0]*(t+surtime)*step[i_t]
            
            #if (N_B[i_s, i_t] < 1.0):
            #    N_B[i_s, i_t] = 1.0
            #if (N_D[i_s, i_t] < 1.0):
            #    N_D[i_s, i_t] = 1.0
            
            #p = poisson.pmf(np.rint(N_B[i_s, i_t]), (f_DM*N_D[i_s, i_t] + N_B[i_s, i_t]))
            #if (p==0):
            #    printf('failed')
            #    printf('baryon event number: ', N_B[i_s, i_t])
            #    printf('dark event number: ', N_D[i_s, i_t])


            #if want_B: rat[i] =  N_D[i]/np.sqrt(N_B[i] + (error*N_B[i])**2) #add 5% inherrent uncertainty on baryon count, so sqrt(N_B + (0.05 N_B)^2)
            #print(t, N_B[i], N_D[i], rat[i])

            i_t = i_t+1
        

        i_s = i_s + 1
        #n = N(m,s['rad'], cutoff, s['glon'], s['glat'],q, rho_dm, baryons, dm_iso = iso, error = error)
        #N_ev = N_ev + n**2
        #print(n**2, N_ev)
        #n_sofar = np.sqrt(N_ev/i)
        #i = i+1
        #print('done for another source')
        #print('done for another source.', n, n_sofar)
        
        
    #print(N_D)
    #print(N_B)
    
    return (N_B, N_D), T

def new_prob(events, f_DM, error = 0.05):
    
    N_B = events[0]
    N_D = events[1]
    
    step = error/5
    Ab = np.arange(1 - 4*error, 1 + 4*error, step)
    def p_alpha(a):
        return norm.pdf(a, loc = 1, scale = error)

    Ptot = 0
    i = 0
    sum_log = np.zeros_like(Ab)
    for ab in Ab:
        Lpois = poisson.logpmf(np.rint(N_B), (f_DM*N_D + ab*N_B))
        #print(np.any(pois==0))
        #print(np.sum(Lpois))
        #print(pois)
        #print(pois)
        #print(np.sum(np.log(pois)))
        sum_log[i] = np.sum(Lpois)
        i = i+1
        #prod = np.prod(pois)
        #print('%f.d10'%prod)
        #Ptot = Ptot + prod*p_alpha(ab)*step
        
    logP = logsumexp(sum_log, b=p_alpha(Ab)*step)
    #print(logP)
    return logP

def max_f(m, N_sources, q, rho_DM, err_baryons = 0.05, logfexp = -4, req_prob = 0.95, data = data, surtime = 4400):
    events, T = gen_events(m, N_sources, q, rho_DM, data = data, surtime = surtime)
    
    print('done generating events')
    
    plogf_array = np.array([])
    logf_array = np.array([])
    
    if (logfexp == 0): #try to dynamically guess the best logfexp
        logfexp = 0
        plogf_m1 = -float('Inf')
        while True:
            f = np.power(10,logfexp)
            P = new_prob(events, f, error = err_baryons)
            plogf = np.log(f) + P #switch to log probability
            #print(logfexp, P, plogf)
            plogf_array = np.append(plogf_array, plogf)
            logf_array = np.append(logf_array, logfexp)
            
            if (plogf < plogf_m1):
                break
                
            if (logfexp < -1000):
                print('failed to find proper spot')
                break
                
            logfexp = logfexp - 0.1
            plogf_m1 = plogf
        
    F = np.logspace(logfexp - 6, logfexp + 6, 120)
    logP = np.zeros_like(F)
    
    #plt.plot(logf_array, plogf_array)
    #plt.show()
    
    #print('events')
    #print(events)
    
    #nplot = 10
    #for i in np.arange(nplot):
    #    plt.plot(T, events[0][i,:], c = [i/nplot, 1 - i/nplot, 0], ls = '-')
    #    plt.plot(T, events[1][i,:], c = [i/nplot, 1 - i/nplot, 0], ls = '--')
    #plt.yscale('log')
    #plt.xscale('log')
    #plt.ylim(1e-2, 1e15)
    #plt.show()
    
    
    i=0
    for f in F:
        lP = new_prob(events, f, error = err_baryons)
        #print(f, lP)
        if(np.isinf(lP)):
            break
        
        logP[i] = lP
        i = i+1
     
    #print('logP: ', logP)
    logP = logP[:i]
    F = F[:i]
    
    #plt.plot(F, F*np.exp(logP))
    #plt.xscale('log')
    #plt.show()
    
    #print(logP)
    #print(logP - np.max(logP))
    
    logP = np.array(logP, dtype = np.float128)
    
    P = np.exp(logP - np.max(logP))
    
    
    Plog = F*P
    
    logF = np.log10(F)
    dif = logF - np.roll(logF, 1)
    dif[0] = dif[1]
    
    normp = np.sum(Plog*dif)
    
    pLnorm = Plog/normp
    
    plt.plot(logF, pLnorm)
    
    
    cs = np.cumsum(pLnorm*dif)
    
    plt.plot(logF, cs)
    plt.show()
    #print(cs[-1])
    #print(cs)
    
    #guess = np.mean(logF)
    #i=0
    #for s in cs:
    #    if(s>req_prob):
    #        guess = logF[i]
    #        break
    #    i = i+1
    
    cs = np.float64(cs)
    fcut = np.interp(req_prob, cs, logF)
    return fcut
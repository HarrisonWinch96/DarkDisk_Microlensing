import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import units as u
from astropy import constants as const

from velfunc import velocity, relative_velocity

import scipy.integrate as integrate

import time


R0 = 8200.0*u.parsec #galactic radius of earth

from density_funcs import *


def galcoord(d, l, b):
    R0 = 8200.0*u.parsec #galactic radius of earth
    l = l*np.pi/180
    b = b*np.pi/180
    z = d*np.sin(b)
    r = np.sqrt(R0**2 + (d*np.cos(b))**2 - 2.0*R0*d*np.cos(b)*np.cos(l))
    th = np.pi - l - np.arcsin(np.sin(l)*R0/r).value #theta = 0 is pointing to earth, theta=pi is ponted away from earth, goes around clockwise
    return r,z,th

def density(x,D,l,b,q,rhofunc):
    d = D*x
    r,z,th = galcoord(d,l,b)
    return rhofunc(r,z,th,q)

def logsense(m,m0,mr):
    return 1.0/(1.0 + (m/m0)**(-1.0/mr))

v_is = 220*u.km/u.s
def v_c(L,x,b,l,iso):
    L = (L/u.pc).value
    #print(L,x,b,l)
    if iso:
        return 220*u.km/u.s
    else:
        return relative_velocity(L,x,b,l,0,0, velocity, True)*u.km/u.s

#R0 = 8200.0*u.parsec #galactic radius of earth

def rE(m, x, L):
    return np.sqrt(4*m*L*x*const.G*(1.0-x)/const.c**2).decompose()

def t_hat(m):
    return 130*np.sqrt(m/u.solMass)*u.day

def argument_starmass(m,L,t,l,b,q,rho_baryon):
    return m**-2.35*integrate.quad(dGdt_arg, 0.0, 1.0, args = (m,L,t,l,b,1.0,rho_baryon))[0]
 
def expfac(m,x,L,t,l,b,iso):
    arg = -4*rE(m,x,L)**2/(t**2*v_c(L,x,b,l,iso)**2).decompose() #velocity is called here
    return np.exp(arg)

def dGdt_arg(x,m,L,t,l,b,q,rhofunc,iso):
    tu = t*u.day
    m = m*u.solMass
    return (u.day**2*32.0*L*density(x,L,l,b,q,rhofunc)*rE(m,x,L)**4*expfac(m,x,L,tu,l,b,iso)/(tu**4*m*v_c(L,x,b,l,iso)**2)).decompose()

def dGdt(t,m,L,cutoff,l,b,q,rho_dm):
    output_DM = integrate.quad(dGdt_arg, 0.0, 1.0, args = (m,L,t,l,b,q,rho_dm))[0]
    
    lowm = np.maximum(0.01,0.01*(t/130)**2)
    highm = np.maximum(1,100*(t/130)**2)
    print('halfway there')
    #output_B = integrate.quad(argument_starmass, lowm, highm, args = (L,t,l,b,q,rho_baryon))[0]
    output_B = integrate.quad(dGdt_arg, 0.0, 1.0, args = (0.36,L,t,l,b,1.0,rho_baryon))[0]
    unc_B = np.sqrt(output_B)
    req_B = unc_B #np.maximum(1.0,unc_B)
    print(t, output_DM, output_B, unc_B, req_B)
    return output_DM*(np.greater(t,cutoff)*np.sqrt(1000.0 +  t))/req_B #used to have *(1000.0 +  t)

def N(m,L,cutoff,l,b,q, rho_dm, want_B, dm_iso = True, surtime = 4400, error = 0):
    L = L*1000*u.parsec #distance to the source, from kpc to pc
    llim = np.log10(np.maximum(cutoff,10*np.sqrt(m)))
    ulim = np.log10(np.maximum(cutoff*1000,10000*np.sqrt(m)))
    #print(llim, ulim)
    n = 30
    T = np.logspace(llim, ulim, n)
    N_B = np.logspace(llim, ulim, n)
    N_D = np.logspace(llim, ulim, n)
    rat = np.logspace(llim, ulim, n)
    step = T - np.roll(T,1)
    step[0] = 0
    i = 0
    for t in T:
        if want_B: N_B[i] = 17e9*integrate.quad(dGdt_arg, 0.0, 1.0, args = (0.36,L,t,l,b,q,rho_baryon, False))[0]*(t+surtime)*step[i] #use average sqrt(mass) insted of average mass
        N_D[i] = 17e9*integrate.quad(dGdt_arg, 0.0, 1.0, args = (m,L,t,l,b,q,rho_dm,dm_iso))[0]*(t+surtime)*step[i]
        if (N_B[i] < 1.0):
            N_B[i] = 1.0
        if (N_D[i] < 1.0):
            N_D[i] = 1.0
            
        
        if want_B: rat[i] =  N_D[i]/np.sqrt(N_B[i] + (error*N_B[i])**2) #add 5% inherrent uncertainty on baryon count, so sqrt(N_B + (0.05 N_B)^2)
        #print(t, N_B[i], N_D[i], rat[i])
        
        i = i+1
    if want_B: 
        chis = np.sqrt(np.sum(rat**2))
    else:
        chis = np.sqrt(np.sum(N_D**2))
    
    #plt.loglog(T, N_B, label = 'baryon')
    #plt.loglog(T,N_D, label = 'DM')
    #plt.loglog(T,np.sqrt(N_B), label = 'sqrt(B)')
    #plt.loglog(T,rat, label = 'ND/sqrt(NB)')
    #plt.loglog(T,step,label = 'stepsize')
    #plt.legend()
    
    return chis

hdu = fits.open('galaxy1.fits')
data0 = hdu[1].data
data_masked = data0[~np.less(np.sqrt(data0['glat']**2 + ((data0['glon']+180)%360 - 180)**2), 2.0)]
data_masked5 = data0[~np.less(np.sqrt(data0['glat']**2 + ((data0['glon']+180)%360 - 180)**2), 5.0)]


from scipy.stats import norm
from scipy.stats import poisson
from scipy.special import logsumexp

def gen_events(m, N_sources, q, rho_dm, data, iso=True, surtime = 4400, baryons = True, m_b = 0.36, random = False):
    cutoff = 1.0
    
    #spots = 0
    datan = 'core'
    
    if (N_sources==1):
        res = 1
        if (data=='LMC'):
            data = np.array([(280.4652, -32.8884, 49.97)], dtype = [('glon', '<f8'), ('glat', '<f8'), ('rad', '<f8')])
            datan = 'LMC'
        else:
            if (data=='core'):
                data = np.array([(0,0,8.2)], dtype = [('glon', '<f8'), ('glat', '<f8'), ('rad', '<f8')])
                datan = 'core'
        
        spots = [0]
        
    else:
        if (random):
            num = np.arange(0,data.size-1)
            spots = np.random.choice(num,N_sources)
            spots = spots.astype(int)
        else:
            #new version to get the right number of sightlines
            spots = np.linspace(0, data.size - 1, N_sources)
            spots = spots.astype(int)
    #i = 1
    
    sources_per_LoS = 17e9/N_sources
    
    llim = np.log10(cutoff)
    #ulim = np.log10(np.maximum(cutoff*100,100000*np.sqrt(m)))
    #ulim = np.log10(np.maximum(cutoff*100,100000*np.sqrt(m)))
    ulim = np.log10(100000000) #used to have two fewer zeros
    #print(llim, ulim)
    n = 30 #used to be 20
    T = np.logspace(llim, ulim, n+1)
    N_B = np.zeros((N_sources, n))
    N_D = np.zeros((N_sources, n))
    N_T = np.zeros((N_sources, n))
    
    rat = np.logspace(llim, ulim, n)
    step = np.roll(T,-1) - T
    step[-1] = 0
    T = T[:-1]
    
    i_s = 0
    
    for s in data[spots]:
        
        L = s['rad']*1000*u.parsec
        l = s['glon']
        b = s['glat']
        
        if (i_s >= N_sources): 
            print('went to far')
            break
        
        i_t = 0
        for t in T:
            
            #if baryons: 
            #    N_B[i_s, i_t] = integrate.quad(dGdt_arg, 0.0, 1.0, args = (m_b,L,t,l,b,q,rho_baryon, False))[0]*(t+surtime)*step[i_t] #use average sqrt(mass) insted of average mass
            #else:
            #    N_B[i_s, i_t] = 0
            
            N_D[i_s, i_t] = integrate.quad(dGdt_arg, 0.0, 1.0, args = (m,L,t,l,b,q,rho_dm, iso))[0]*(t+surtime)*step[i_t]
            
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
        
    if (baryons):
        if (N_sources==1):
            try:
                if (datan=='LMC'):
                    N_B = np.loadtxt('baryon_events_LMC.txt')
                else:
                    if (datan=='core'):
                        N_B = np.loadtxt('baryon_events_core.txt')
            except:
                print('cant load which single sightline, defaulting to core')
                N_B = np.loadtxt('baryon_events_core.txt')
        else:   
            name = 'baryon_events_LoS_{}.txt'.format(str(N_sources))
            try:
                N_B = np.loadtxt(name)
            except:
                print("Haven't run the baryon events for {} LoS.".format(str(N_sources)))
                print('defaulting to 100 LoS, but will probably break')
                N_B = np.loadtxt('baryon_events_LoS_100.txt')
    else:
        N_B = np.zeros_like(N_D)
            
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

def new_prob(events, f_DM, N_sources, error = 0.05, baryons = True):
    
    N_D = events[1]
    
    N_B = events[0]
    #if (baryons):
    #    N_B = np.loadtxt('baryon_events.txt')
    #else:
    #    N_B = np.zeros_like(N_D)
    
    
    
    step = error/5
    
    llim = np.maximum(1 - 4*error, 0)
    Ab = np.arange(llim, 1 + 4*error, step)
    def p_alpha(a):
        return norm.logpdf(a, loc = 1, scale = error)

    sources_per_LoS = 17e9/N_sources
    
    Ptot = 0
    i = 0
    sum_log = np.zeros_like(Ab)
    
    k = np.expand_dims(np.expand_dims(np.arange(0,100), 0), 0)

    N_D = np.expand_dims(N_D, 2)
    N_B = np.expand_dims(N_B, 2)
        
        
    for ab in Ab:
        
        karray = poisson.pmf(k, N_B)*np.nan_to_num(poisson.logpmf(k, (f_DM*N_D + ab*N_B)))
        Lpois = sources_per_LoS*np.sum(karray,2)
        #print(np.any(pois==0))
        #print(np.sum(Lpois))
        #print(pois)
        #print(pois)
        #print(np.sum(np.log(pois)))
        sum_log[i] = np.sum(Lpois) + p_alpha(ab) + np.log(step)
        i = i+1
        #prod = np.prod(pois)
        #print('%f.d10'%prod)
        #Ptot = Ptot + prod*p_alpha(ab)*step
        
    logP = logsumexp(sum_log)#, b=p_alpha(Ab)*step)
    #print(logP)
    return logP

def max_f(m, N_sources, q, rho_DM, err_baryons = 0.05, logfexp = 0, req_prob = 0.95, data = data0, surtime = 4400, iso = True, baryons = True, random = False, m_b = 0.36):
    events, T = gen_events(m, N_sources, q, rho_DM, data, iso = iso, surtime = surtime, m_b = m_b, baryons = baryons, random = random)
    #print(events)
    print('done generating events')
    
    plogf_array = np.array([])
    logf_array = np.array([])
    
    if (logfexp == 0): #try to dynamically guess the best logfexp
        logfexp = 0
        break_counter = 0
        plogf_m1 = -float('Inf')
        while True:
            f = np.power(10,logfexp)
            P = new_prob(events, f, N_sources, error = err_baryons, baryons = baryons)
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
    
    print('found where to start')
    
    i=0
    for f in F:
        lP = new_prob(events, f, N_sources, error = err_baryons)
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
    
    #plt.plot(logF, pLnorm)
    
    
    cs = np.cumsum(pLnorm*dif)
    
    #plt.plot(logF, cs)
    #plt.show()
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
from ML_functions import max_f, logsense
from density_funcs import *
import numpy as np
#import matplotlib.pyplot as plt


def basic_constraints():
    M = np.logspace(-4, 7, num = 20)
    fmd = np.zeros_like(M)
    lfmd = np.zeros_like(M)

    i = 0
    for m in M:
        lfmd[i] = max_f(m, 20, 1.0, rho_NFW, logfexp = 0) #use the new function
        fmd[i] = np.power(10, lfmd[i])
        print(m, lfmd[i], fmd[i])
        i = i+1
    
    #plt.loglog(M,fmd/logsense(M,1e-2,0.5))
    #plt.savefig('test.png')
    np.savetxt('test.txt', fmd)
    return fmd

def baryon_errors(error):
    M = np.logspace(-4, 7, num = 20)

    m_array = np.array([])
    n_20_bar_array = np.array([])
    
    cutoff = 1
    nstars = 1.0e9

    for m in M:
        #start = time.time()
        n_20_bar = max_f(m,100,1.0, rho_NFW, err_baryons = error)
        

        m_array = np.append(m_array, m)
        n_20_bar_array = np.append(n_20_bar_array, n_20_bar)
        
        #end = time.time()
        #tim = end - start
        print(m,n_20_bar_array)

    #np.savetxt('m_array.txt', m_array)
    name = "n_100_error_{}.txt".format(error)
    np.savetxt(name, n_20_bar_array)
    
    
def runtime():
    M = np.logspace(-4, 7, num = 20)

    
    core_nobar_array = np.array([])

    core_nobar_notime_array = np.array([])
    core_nobar_1yr_array = np.array([])

    print('started runtime')

    for m in M:
        #start = time.time()
        
        core_nobar = max_f(m, 100, 1.0, rho_NFW, surtime = 4400, baryons = False)
        core_nobar_notime = max_f(m, 100, 1.0, rho_NFW, surtime = 0, baryons = False)
        core_nobar_1yr = max_f(m, 100, 1.0, rho_NFW, surtime = 365, baryons = False)
        #print('done the others')

        #plt.scatter(m,n, c = 'g')
        #result_array_20los = np.append(result_array_20los, result)

        
        core_nobar_array = np.append(core_nobar_array, core_nobar)

        core_nobar_notime_array = np.append(core_nobar_notime_array, core_nobar_notime)
        core_nobar_1yr_array = np.append(core_nobar_1yr_array, core_nobar_1yr)

        #end = time.time()
        #tim = end - start
        #print(m,n_20_bar,tim)
        print('done for ', m)

    np.savetxt('time_fulltime.txt', core_nobar_array)
    np.savetxt('time_notime.txt', core_nobar_notime_array)
    np.savetxt('time_1yr.txt', core_nobar_1yr_array)

#print('got to running the func ok')
#basic_constraints()


def source_bar_compare():
    M = np.logspace(-4, 7, num=40)
    
    core_nobar_array= np.array([])
    core_bar_array = np.array([])
    lmc_nobar_array = np.array([])
    lmc_bar_array = np.array([])
    n20_nobar_array = np.array([])
    n20_bar_array = np.array([])
    
    print('start runtime')
    
    for m in M:
        core_nobar = max_f(m, 1, 1.0, rho_NFW, data='core', baryons = False)
        core_bar = max_f(m, 1, 1.0, rho_NFW, data='core', baryons = True)
        lmc_nobar = max_f(m, 1, 1.0, rho_NFW, data='LMC', baryons = False)
        lmc_bar = max_f(m, 1, 1.0, rho_NFW, data='LMC', baryons = True)
        n20_nobar = max_f(m, 100, 1.0, rho_NFW, baryons = False)
        n20_bar = max_f(m, 100, 1.0, rho_NFW, baryons = True)
        
        core_nobar_array = np.append(core_nobar_array, core_nobar)
        core_bar_array = np.append(core_bar_array, core_bar)
        lmc_nobar_array = np.append(lmc_nobar_array, lmc_nobar)
        lmc_bar_array = np.append(lmc_bar_array, lmc_bar)
        n20_nobar_array = np.append(n20_nobar_array, n20_nobar)
        n20_bar_array = np.append(n20_bar_array, n20_bar)
        
        print('source bar compare, done for ', m)
        
    np.savetxt('sbc_core_nobar.txt', core_nobar_array)
    np.savetxt('sbc_core_bar.txt', core_bar_array)
    np.savetxt('sbc_lmc_nobar.txt', lmc_nobar_array)
    np.savetxt('sbc_lmc_bar.txt', lmc_bar_array)
    np.savetxt('sbc_n20_nobar.txt', n20_nobar_array)
    np.savetxt('sbc_n20_bar.txt', n20_bar_array)

    
def sphericality(q):
    N = np.array([5,20,100])
    #fn = np.zeros_like(N)
    m=1.0
    f_array = np.zeros(5)
    
    f_array[0] = max_f(m, 1, q, rho_NFW, data='core')
    f_array[1] = max_f(m, 1, q, rho_NFW, data='LMC')
    
    print('done core and LMC for q = ', q)
    
    
    i=2
    #for n in N:
    f_array[i] = max_f(m, 100, q, rho_NFW)
    #    i = i+1
        
    print('all done q = ', q)
    
    np.savetxt('sphericality_{}.txt'.format(str(q)), f_array)
    
def rescaled_disk(m=1.0):
    Qr = np.logspace(-2,2,num = 17)
    Qz = np.logspace(-2,2,num = 17)
    #Q = np.meshgrid(Qr, Qr)
    f_array = np.zeros([17,17])

    ir = 0
    for qr in Qr:
        iz = 0
        for qz in Qz:
            q = np.array([qr,qz])
            f = max_f(m, 100, q, rho_mirrordisk)
            f_array[ir,iz] = f
            iz = iz+1
        ir = ir+1
    
    np.savetxt('rescaled_disk_m{}.txt'.format(str(m)), f_array)
    
    
def DD_mass_constraints(q=1.0):
    qa = np.array([q,q])
    n=20
    M = np.logspace(-4, 7, num = n)
    f = np.zeros_like(M)
    j = 0
    for m in M:
        logf = max_f(m, 100, qa, rho_mirrordisk, logfexp = 0) #use the new function
        print('Done for q, m: ', q, m)
        f[j] = logf
        #print(j, m, f[i,j])
        j = j+1
        
    np.savetxt('rescaled_DD_mass_q{}.txt'.format(str(q)), f)
    
    
def tiltdisk(m=1.0):
    
    Phis = np.linspace(np.pi/16, np.pi/2, 8)
    Thetas = np.arange(0,2*np.pi, np.pi/2)

    result_array_tilt = np.zeros([8,4])

    j = 0
    for theta in Thetas:
        i = 0
        for phi in Phis:
            q = [theta,phi]
            n_50 = max_f(m,100, q, rho_mirrordisk_tilt)
            result_array_tilt[i,j] = n_50
            print('tiltdisk finished for ', theta, phi, n_50)
            i = i+1
        j = j+1

    rat0 = max_f(m, 100, [0,0], rho_mirrordisk_tilt)
    
    rat = np.append(rat0*np.ones([1,4]), result_array_tilt, axis = 0)
    print('tiltdisk finished for m=', m)
    print(rat)
    
    np.savetxt('result_array_tiltdisk_{}.txt'.format(str(m)), rat)
    
    
def n_los(m, q, rho, name, rand = False):
    
    N = np.array([5, 10, 20, 50, 100, 200, 500, 1000, 2000])
    
    f_array = np.zeros([2+9*3])
    
    f_array[0] = max_f(m, 1, q, rho, data='core')
    f_array[1] = max_f(m, 1, q, rho, data='LMC')
    
    from astropy.io import fits

    hdu = fits.open('galaxy1.fits')
    data0 = hdu[1].data
    data_masked = data0[~np.less(np.sqrt(data0['glat']**2 + ((data0['glon']+180)%360 - 180)**2), 2.0)]
    data_masked5 = data0[~np.less(np.sqrt(data0['glat']**2 + ((data0['glon']+180)%360 - 180)**2), 5.0)]

    
    i = 2
    for n in N:
        for j in np.arange(3):
            f = max_f(m, n, q, rho, data=data0, random = True)
            f_array[i] = f
            i = i+1
        
    #for n in N:
    #    f = max_f(m, n, q, rho, data=data_masked)
    #    f_array[i] = f
    #    i = i+1
    #    
    #for n in N:
    #    f = max_f(m, n, q, rho, data=data_masked5)
    #    f_array[i] = f
    #    i = i+1
    
    np.savetxt('n_LoS_m{}_testrand.txt'.format(name), f_array)
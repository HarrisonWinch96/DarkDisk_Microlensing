import numpy as np
from astropy import units as u

def rho_baryon(r,z,th,q):
    A = 0.04*u.solMass/(u.parsec**3)
    hr = 3000.0*u.parsec
    R0 = 8200.0*u.parsec
    hz = 400.0*u.parsec
    return A*np.exp(-1*(r - R0)/hr)*np.exp(-1*np.abs(z)/hz)

def rho_mirrordisk(r,z,th,q): #dark version of the baryonic disk
    #use q as 2-element array to rescale z and r
    #figure out normalization so total DM stays the same
    #maybe make fiducial model have same mass as baryons, not CDM
    #so instead of f_DM, we would be constraining rho_mir/rho_bar
    A = 0.04*u.solMass/(u.parsec**3) /(q[0]**2*q[1])
    hr = 3000.0*u.parsec
    R0 = 8200.0*u.parsec
    hz = 400.0*u.parsec
    return A*np.exp(-1*(r/q[0] - R0)/hr)*np.exp(-1*np.abs(z/q[1])/hz)
    
    
def rho_mirrordisk_tilt(r,z,th,q): #tilted by some angles, q = [theta of tilt, +phi]
    A = 0.04*u.solMass/(u.parsec**3)
    hr = 3000.0*u.parsec 
    R0 = 8200.0*u.parsec
    hz = 400.0*u.parsec
    
    theta = q[0]
    phi = q[1]
    x,y,z = r*np.cos(th), r*np.sin(th), z
    
    x,y,p = x*np.cos(theta) - y*np.sin(theta), y*np.cos(theta) + x*np.sin(theta), z #rotate theta around z axis
    
    x,y,z = x, y*np.cos(phi) - z*np.sin(phi), z*np.cos(phi) + y*np.sin(phi)
    
    x,y,p = x*np.cos(theta) + y*np.sin(theta), y*np.cos(theta) - x*np.sin(theta), z #rotate theta around z axis
    
    
    r = np.sqrt(x**2 + y**2)
    z = z
    return A*np.exp(-1*(r - R0)/hr)*np.exp(-1*np.abs(z)/hz)
    
def rho_semis(r, z,th,q): #standard semi-isothermal halo
    R = np.sqrt(r**2 + (z/q)**2)
    A = 0.01 *u.solMass / (u.parsec**3) #originally 0.0079
    R0 = 8200.0*u.parsec
    Rc = 5000.0*u.parsec
    #print(r,z,R)
    return A*(R0**2 + Rc**2)/(q*(R**2 + Rc**2))

def rho_NFW(r,z,th,q): #NFW halo distribution
    A = 0.014*u.solMass / (u.parsec**3)
    Rs = 16000*u.parsec
    
    R = np.sqrt(r**2 + (z/q)**2)
    
    x = R/Rs    
    return A/(q*x*(1 + x)**2)

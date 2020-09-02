from galpy.orbit import Orbit
from galpy.potential import MWPotential2014
from galpy.potential import plotRotcurve, vcirc
from galpy.actionAngle import actionAngleAdiabatic
from galpy.df import quasiisothermaldf
import numpy as np
from math import sqrt, sin, cos, tan, atan2, exp
from random import seed, random
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import interactive
from astropy import units as u
from astropy import constants as const
interactive(True)
 
PI = 3.141592
 
seed(datetime.now())
 
mp = MWPotential2014
R_0 = 8200.0
 
def random_direction():
    # returns a random 3D unit vector
    x1 = x2 = 1000.
    while (x1*x1 + x2*x2 >= 1):
        x1 = 2*random()-1
        x2 = 2*random()-1
    return np.array([2*x1*sqrt(1-x1*x1-x2*x2),2*x2*sqrt(1-x1*x1-x2*x2),1-2*(x1*x1+x2*x2)])
 
def v_orbit(r):
    return 220.0*vcirc(mp, r/R_0)
 
def v_disp_r(r):
    return 35.0*exp(-(r-R_0)/R_0)
 
def v_disp_z(r):
    return 20.0*exp(-(r-R_0)/R_0)
 
def sample(sigma):
    return np.random.normal(0.0, sigma)
 
r_sun = R_0
# velocity of the sun:
v_sun = np.array([0, v_orbit(r_sun), 0])
# position of the sun:
x_sun = np.array([r_sun,0,0])
 
def velocity(distance, theta_s, phi_s, theta_d, phi_d, disperse=True):
    # distance is distance of object from the sun
    # theta_s and phi_s are not quite galactic coordinates, theta_s=0 points OUT of the galaxy
    # theta_d and phi_d denote the direction of the normal vector of the plane of the dark disk
    # set disperse to True or False depending on whether you want to sample from the velocity distribution or just get the average
    # line of sight from the sun to the object:
    los = np.array([sin(theta_s)*cos(phi_s), sin(theta_s)*sin(phi_s), cos(theta_s)])
    # position of the object
    x = (-1)*x_sun + distance*los
    r = sqrt(np.dot(x,x))
    u = np.array([  distance*( cos(theta_s)*sin(theta_d)*sin(phi_d) - cos(theta_d)*sin(theta_s)*sin(phi_s) ),
                    -distance*cos(theta_s)*cos(phi_d)*sin(theta_d) + cos(theta_d)*( -r_sun + distance*cos(phi_s)*sin(theta_s) ),
                    sin(theta_d)*( (r_sun - distance*cos(phi_s)*sin(theta_s) )*sin(phi_d) + distance*cos(phi_d)*sin(theta_s)*sin(phi_s) ) ])
    velocity_direction = (-1)*u/(sqrt(np.dot(u,u)))
    if disperse:
        return sqrt(v_orbit(r)**2 + v_disp_z(r)**2 + v_disp_r(r)**2)*velocity_direction
    else:
        return v_orbit(r)*velocity_direction
 
def relative_velocity(d, p, b, l, theta_d, phi_d, velocity, disperse=True):
    # d is distance to the source
    # p is fractional distance of the lens
    # b, l is galactic latitude, longitude
    # psi_source, psi_lens are the angles the velocities of the source/lens make to the galactic plane
    # set disperse to True or False depending on whether you want to sample from the velocity distribution or just get the average
    # line of sight to the source:
    los = np.array([sin(PI/2-b)*cos(l), sin(PI/2-b)*sin(l), cos(PI/2-b)])
    # velocity of the source in sun's rest frame:
    #print(d)
    v_source = velocity(d,PI/2-b,l,theta_d,phi_d,disperse) - v_sun
    # velocity of the lens in sun's rest frame:
    v_lens = velocity(d*p,PI/2-b,l,theta_d,phi_d,disperse) - v_sun
    # component of source velocity perpendicular to line of sight:
    v_source_perp = v_source - np.dot(v_source,los)*los
    # component of lens velocity perpendicular to line of sight:
    v_lens_perp = v_lens - np.dot(v_lens,los)*los
    return np.linalg.norm(v_source_perp - v_lens_perp)
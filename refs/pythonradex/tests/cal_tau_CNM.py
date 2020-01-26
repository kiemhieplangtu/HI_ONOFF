# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 11:07:35 2017

@author: gianni
"""

# from pythonradex import atomic_transition
import sys
from scipy import constants
import pytest
import numpy as np
import matplotlib.pyplot as plt


def tau_nu(nu, nu0, NHI, Tex, phi_nu, A21=2.85e-15):
    '''Compute the optical depth from the column densities N1 and N2 in the lower
    and upper level respectively.'''
    # return (constants.c**2/(8*np.pi*nu**2) * A21 * phi_nu(nu) * (g2/g1*N1-N2))
    return (3.*constants.c**2/(32.*np.pi*nu) * A21 * phi_nu * NHI * constants.h/(constants.k*Tex) )

def tau_vel(nu0, NHI, Tex, phi_v, A21=2.85e-15):
    return (3.*constants.c**3/(32.*np.pi*nu0**2) * A21 * phi_v * NHI * constants.h/(constants.k*Tex) )

def FWHM2sigma(FWHM):
    """Convert FWHM of a Gaussian to standard deviation.
    
    Parameters
    -----------
    FWHM: float or numpy.ndarray
        FWHM of the Gaussian
    
    Returns
    ------------
    float or numpy.ndarray
        the standard deviation of the Gaussian"""
    return FWHM/(2.*np.sqrt(2.*np.log(2.)))

def cal_phi_nu(nu,nu0,width_nu):
    '''compute the normalised line profile for the frequency nu'''
    sigma_nu = FWHM2sigma(width_nu)
    norm = 1./(sigma_nu*np.sqrt(2.*np.pi))
    # norm = 1.
    return norm*np.exp(-(nu-nu0)**2/(2.*sigma_nu**2))


def phi_v(v, v0, width_v):
    sigma_v = FWHM2sigma(width_v)
    norm = 1./(sigma_v*np.sqrt(2.*np.pi))
    return norm*np.exp(-(v-v0)**2/(2.*sigma_v**2))


def freq_range(Delta_v,nu0):
    '''Computes the frequency interval from a given velocity interval
    Delta_v at frequency nu0'''
    return nu0 * Delta_v / constants.c


def freq2vel(nu,nu0):
    return (nu0-nu)*constants.c/nu0



nu0     = 1.42*constants.giga
v0      = 4.47*constants.kilo
width_v = 3.*constants.kilo
nchan   = 1024
test_v  = np.linspace(-5.*width_v, 5.*width_v, nchan)
dv      = 10.*width_v/nchan

nu = nu0*(1.-test_v/constants.c)
wid_nu = freq_range(width_v,nu0)
phi_nu = cal_phi_nu(nu,nu0,wid_nu)

phi_v = phi_v(test_v, v0, width_v)

if(False):
	plt.plot(test_v/constants.kilo, phi_v)
	plt.show()


print 'np.trapz(phi_v, dx=dv)', np.trapz(phi_v, dx=dv)


A21 = 2.85e-15 # s^-1
g2 = 3
g1 = 1


# I assumed the following column density (typical values from your paper) :
# WNM(front)=WNM(back)=5E20 cm^-2.

# CNM = 2E20 cm^-2

# The WNM is always at 8000K.

# The spin temperature of the CNM is drawn from a log-normal distribution that looks like the one you obtained (Fig. 24, grey line). 

# NCNM = 10.e20 # 2E20 cm^-2
NCNM = 1.e20 # 2E20 cm^-2
Tex = 80.
# Tex = 50.8
# Tc = 100.
# Tc = 2.725
Tbg = 5.15
Tc = 100000.


NCNM = NCNM*1.e4 ## To m^-2  http://www.ucolick.org/~xavier/AY230/ay230_HI21cm.pdf

tau_nu = tau_nu(nu, nu0, NCNM, Tex, phi_nu, A21=A21)

# vel = freq2vel(nu_array,nu0)
tau_vel = tau_vel(nu0, NCNM, Tex, phi_v)

## Add noise
# noise_mean  = 0.
# noise_sigma = 0.005
# noise       = np.random.normal(noise_mean, noise_sigma, len(phi_v))
# tau_vel     = tau_vel # + noise

print 'N(HI)', 1.e-4 * np.trapz(tau_vel, dx=dv)*Tex*32.*np.pi*nu0**2*constants.k / (3.*constants.c**3 * A21 * constants.h)


# Toff_vel = Tbg*np.exp(-tau_vel) + Tex*(1. - np.exp(-tau_vel))
Toff_vel = Tex*(1. - np.exp(-tau_vel))
Ton_vel  = (Tbg+Tc)*np.exp(-tau_vel) + Tex*(1. - np.exp(-tau_vel))

print 'example Tau_peak: ', 1.e20 / 1.825e18 / 80. / 3./ np.sqrt(2.*np.pi)
print 'Max Tau:', np.max(tau_vel)
print 'Max Tau_nu:', np.max(tau_nu)

if(False):
	plt.plot(test_v/1000., tau_vel)
	# plt.plot(test_v/1000., Toff_vel)
	plt.ylabel('Tau')
	plt.show()


	plt.plot(nu, tau_nu)
	# plt.plot(test_v/1000., Toff_vel)
	plt.ylabel('Tau')
	plt.show()

	sys.exit()


print np.max(tau_vel)


fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10,8))
axs[0].plot(test_v/1000., Toff_vel-Tbg)
# axs[0].plot(test_v/1000., tau_vel)
axs[0].set_ylabel('T_OFF')
# axs[0].set_xlim(0, 2)
# axs[0].grid(True)

# axs[1].plot(test_v/1000., np.exp(-tau_vel))
axs[1].plot(test_v/1000., tau_vel)
axs[1].set_ylabel('e(-tau')
axs[1].set_xlabel('Vel')

fig.tight_layout()
plt.show()
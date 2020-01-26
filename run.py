import sys
from scipy import constants
import numpy as np
import matplotlib.pyplot as plt


def tau_nu(nu, nu0, NHI, Tex, phi_nu, A21=2.85e-15):
    return (3.*constants.c**2/(32.*np.pi*nu) * A21 * phi_nu * NHI * constants.h/(constants.k*Tex) )





def tau_vel(nu0, NHI, Tex, phi_v, A21=2.85e-15):
    nr, nc = phi_v.shape    
    ret    = np.zeros( (nr, nc) )
    for i in range(nr):
        ret[i,:] = (3.*constants.c**3/(32.*np.pi*nu0**2) * A21 * phi_v[i,:] * NHI[i] * constants.h/(constants.k*Tex[i]) )
    return ret




def FWHM2sigma(FWHM):
    return FWHM/(2.*np.sqrt(2.*np.log(2.)))





def phi_nu(nu,nu0,wid_nu):
    sigma_nu = FWHM2sigma(width_nu)
    norm     = 1./(sigma_nu*np.sqrt(2.*np.pi))
    # norm = 1.
    return norm*np.exp(-(nu-nu0)**2/(2.*sigma_nu**2))




def phi_vel(v, v0, wid_v):
    sigma_v = FWHM2sigma(wid_v)
    norm    = 1./(sigma_v*np.sqrt(2.*np.pi))
    
    nr  = len(v0)
    nc  = len(v)
    
    ret = np.zeros( (nr, nc) )
    for i in range(nr):
        ret[i,:] = norm[i]*np.exp(-(v-v0[i])**2/(2.*sigma_v[i]**2))
    return ret




def dv2dnu(Delta_v,nu0):
    return nu0 * Delta_v / constants.c




def freq2vel(nu,nu0):
    return (nu0-nu)*constants.c/nu0







def cal_Toff_WNM(tau, Tbg, Tex):
    nr, nc = tau.shape
    ret    = np.zeros( (nr, nc) )
    for i in range(nr):
        ret[i,:] = Tbg*np.exp(-tau[i,:]) + Tex[i]*(1. - np.exp(-tau[i,:]))

    return ret





def tb_exp(tau, Tbg, Tex, ncnm, Toff_wnm, Tex_wnm, fwnm, nwnm, ordercnm=[0,1]):
    nrcnm, nchnl = tau.shape

    #FIRST STEP IS TO CALCULATE THE OPACITY OF EACH COLD CLOUD...
    # taucnmxx = np.zeros( (nrcnm, nchnl), dtype='float64' )

    if(nrcnm != 1):
        tausum = np.sum( tau, 0) 
    else:
        tausum = tau[0]

    exp_tausum = np.exp( -tausum )
    tb_cont    = Tbg * exp_tausum


    # ********** NEXT CALCULATE THE WNM CONTRIBUTION ********************
    # WE EXPRESS THE WNM CONTRIBUTION AS A SUM OF GAUSSIANS:
    #   FWNM, Tbg, HGTWNM, CENWNM, WIDWNM

    # THESE ARE SELF-EXPLANATORY EXCEPT FOR FWNM. 
    # 
    # THE CONTRIBUTION OF THE WNM COMPONENTS TO THE ANTENNA TEMP IS...
    # WNM CONTRIBUTION  = SUM, k FROM 0 TO K { G_WNMk [f_k + (1-f_k)exp(-tausum)]}
    # WHERE TAUSUM IS THE SUM OF THE OPACITIES OF ALL THE CLOUDS.

    tb_wnm_tot = np.zeros(nchnl, dtype='float64' )
    for i in range(nwnm):
        tb_wnm_nrw = Toff_wnm[i]
        tb_wnm_tot = tb_wnm_tot + tb_wnm_nrw*( fwnm[i] + (1.- fwnm[i])*exp_tausum )

    # *************** NEXT CALCULATE THE CNM CONTRIBUTION ****************
    tb_cnm_tot = np.zeros( nchnl, dtype='float64' )

    for nrc in range(nrcnm):
        tausum_nrc  = np.sum( tau[0:nrc, :], 0)
        exp_tau_nrc = np.exp( -tausum_nrc )
        tb_cnm_tot  = tb_cnm_tot + Tex[nrc] * (1. - np.exp(-tau[nrc, :]) ) * exp_tau_nrc
    ## Endfor

    tb_tot = tb_cont + tb_cnm_tot + tb_wnm_tot

    return tb_tot, tb_cont, tb_cnm_tot, tb_wnm_tot, tausum






def addnoise(nr, nc, noise_mean = 0., noise_sigma = 0.025):
    return np.random.normal(noise_mean, noise_sigma, [nr, nc])

##=========================================================================================#





#### MAIN ####
'''
04 HI clouds along a LOS: Source -> WNM -> CNM -> CNM -> WNM -> observers
'''

## HI 21 cm 
nu0    = 1.42*constants.giga

## Vel range
test_v = np.linspace(-30.*constants.kilo, 30.*constants.kilo, 2048)
dv     = 60.*constants.kilo/2047.


## 02 CNM clouds
v0    = [0.5, 4.47]
v0    = np.array(v0)*constants.kilo

wid   = [3.2, 2.06]
wid   = np.array(wid)*constants.kilo

ncnm  = len(v0)
nchan = len(test_v)

phi_v = phi_vel(test_v, v0, wid)


print ('np.trapz(phi_v, dx=dv)', np.trapz(phi_v, dx=dv))


A21     = 2.85e-15 # s^-1
g2      = 3
g1      = 1

NCNM    = np.array([1.e20, 2.23e20]) # 2E20 cm^-2
Tex     = np.array([72., 50.8])
Tbg     = 5.15
Tc      = 100000.


NCNM    = NCNM*1.e4 ## To m^-2  
tau_cnm = tau_vel(nu0, NCNM, Tex, phi_v) + addnoise(ncnm, nchan) # http://www.ucolick.org/~xavier/AY230/ay230_HI21cm.pdf



## 02 WNM clouds
v0       = [1.5, 10.5]
v0       = np.array(v0)*constants.kilo

wid      = [9.5, 8.]
wid      = np.array(wid)*constants.kilo

fwnm     = [1., 0.]
fwnm     = np.array(fwnm)

nwnm     = len(v0)

phi_v    = phi_vel(test_v, v0, wid)

NWNM     = np.array([5.e20, 8.e20]) # 2E20 cm^-2
Tex_wnm  = np.array([8500., 8000.])


NWNM     = NWNM * 1.e4 ## To m^-2
tau_wnm  = tau_vel(nu0, NWNM, Tex_wnm, phi_v) + addnoise(nwnm, nchan, noise_mean = 0., noise_sigma = 0.0001) ## This may be crazy, but that's ok :)
Toff_wnm = cal_Toff_WNM(tau_wnm, Tbg, Tex_wnm)



## Cal Texp [+]
tb_tot, tb_cont, tb_cnm_tot, tb_wnm_tot, tausum = tb_exp(tau_cnm, Tbg, Tex, ncnm, Toff_wnm, Tex_wnm, fwnm, nwnm) # cal_Toff(tau_vel, Tbg, Tex)





## Plots
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10,8))
axs[0].plot(test_v/1000., tb_tot-Tbg)
axs[0].set_ylabel('OFF: Texp [K]')

axs[1].plot(test_v/1000., np.exp(-tausum))
axs[1].set_ylabel('ON: e(-tau)')
axs[1].set_xlabel('V [km/s]')

fig.tight_layout()
plt.show()
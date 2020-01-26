# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 11:07:35 2017

@author: gianni
"""

from pythonradex import atomic_transition
from scipy import constants
import pytest
import numpy as np
import matplotlib.pyplot as plt


class TestLineProfile():
    nu0 = 400*constants.giga
    width_v = 10*constants.kilo
    gauss_line_profile = atomic_transition.GaussianLineProfile(nu0=nu0,width_v=width_v)
    square_line_profile = atomic_transition.SquareLineProfile(nu0=nu0,width_v=width_v)
    profiles = (gauss_line_profile,square_line_profile)
    test_v = np.linspace(-3*width_v,3*width_v,600)

    def test_abstract_line_profile(self):
        with pytest.raises(NotImplementedError):
            atomic_transition.LineProfile(nu0=self.nu0,width_v=self.width_v)
    
    def test_constant_average_over_nu(self):
        for profile in self.profiles:
            const_array = np.ones_like(profile.nu_array)
            const_average = profile.average_over_nu_array(const_array)
            assert np.isclose(const_average,1,rtol=1e-2,atol=0)
    
    def test_asymmetric_average_over_nu(self):
        for profile in self.profiles:
            left_value,right_value = 0,1
            asymmetric_array = np.ones_like(profile.nu_array)*left_value
            asymmetric_array[:asymmetric_array.size//2] = right_value
            asymmetric_average = profile.average_over_nu_array(asymmetric_array)
            assert np.isclose(asymmetric_average,np.mean((left_value,right_value)),
                              rtol=1e-2,atol=0)
    
    def test_square_profile_average_over_nu(self):
        np.random.seed(0)
        nu_array = self.square_line_profile.nu_array
        random_values = np.random.rand(nu_array.size)
        profile_window = np.where(self.square_line_profile.phi_nu(nu_array)==0,0,1)
        expected_average = np.sum(profile_window*random_values)/np.count_nonzero(profile_window)
        average = self.square_line_profile.average_over_nu_array(random_values)
        assert np.isclose(expected_average,average,rtol=5e-2,atol=0)
    
    def test_normalisation(self):
        for profile in self.profiles:
            integrated_line_profile = np.trapz(profile.phi_nu_array,profile.nu_array)
            integrated_line_profile_v = np.trapz(profile.phi_v(self.test_v),self.test_v)
            for intg_prof in (integrated_line_profile,integrated_line_profile_v):
                 assert np.isclose(intg_prof,1,rtol=1e-2,atol=0)

    def test_profile_shape(self):
        square_phi_nu = self.square_line_profile.phi_nu_array
        square_phi_v = self.square_line_profile.phi_v(self.test_v)
        for square_phi,x_axis,width in zip((square_phi_nu,square_phi_v),
                                     (self.square_line_profile.nu_array,self.test_v),
                                     (self.square_line_profile.width_nu,self.width_v)):
            assert square_phi[0] ==  square_phi[-1] == 0
            assert square_phi[square_phi.size//2] > 0
            square_indices = np.where(square_phi>0)[0]
            square_window_size = x_axis[square_indices[-1]] - x_axis[square_indices[0]]
            assert np.isclose(square_window_size,width,rtol=5e-2,atol=0)
        gauss_phi_nu = self.gauss_line_profile.phi_nu_array
        gauss_phi_v = self.gauss_line_profile.phi_v(self.test_v)
        for gauss_phi,x_axis,width in zip((gauss_phi_nu,gauss_phi_v),
                                          (self.square_line_profile.nu_array,self.test_v),
                                          (self.square_line_profile.width_nu,self.width_v)):
            assert np.all(np.array((gauss_phi[0],gauss_phi[-1]))
                          <gauss_phi[gauss_phi.size//2])
            max_index = np.argmax(gauss_phi)
            half_max_index = np.argmin(np.abs(gauss_phi-np.max(gauss_phi)/2))
            assert np.isclose(2*np.abs(x_axis[max_index]-x_axis[half_max_index]),
                              width,rtol=3e-2,atol=0)



nu0                 = 400*constants.giga
width_v             = 10*constants.kilo
gauss_line_profile  = atomic_transition.GaussianLineProfile(nu0=nu0,width_v=width_v)
square_line_profile = atomic_transition.SquareLineProfile(nu0=nu0,width_v=width_v)
profiles            = (gauss_line_profile,square_line_profile)
test_v              = np.linspace(-3*width_v,3*width_v,600)

nu = nu0*(1.-test_v/constants.c)

print nu0
print width_v
print gauss_line_profile.sigma_nu
print gauss_line_profile.normalisation

phi_v = gauss_line_profile.phi_v(test_v)
phi_nu = gauss_line_profile.phi_nu(nu)
print phi_v

# plt.plot(test_v, phi_v)
# plt.show()

plt.plot(nu, phi_nu)
plt.show()
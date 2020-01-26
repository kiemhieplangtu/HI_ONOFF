# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 11:07:35 2017

@author: gianni
"""

from pythonradex import atomic_transition
from scipy import constants
import pytest
import numpy as np


class TestLevel():

    g = 2.
    E = 3.  # energy in [J]
    number = 1
    level = atomic_transition.Level(g=g,E=E,number=number)
    
    def test_LTE_level_pop(self):
        T = 50
        Z = 3
        lte_level_pop = self.level.LTE_level_pop(Z=Z,T=T)
        assert lte_level_pop == self.g*np.exp(-self.E/(constants.k*T))/Z
        shape = (5,5)
        T_array = np.ones(shape)*T
        Z_array = np.ones(shape)*Z
        lte_level_pop_array = self.level.LTE_level_pop(Z=Z_array,T=T_array)
        assert lte_level_pop_array.shape == shape
        assert np.all(lte_level_pop==lte_level_pop_array)



test = TestLevel()
print test.g
print test.E
print test.number
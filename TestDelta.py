#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 11:14:23 2017

@author: jacob
"""
import numpy as np
import matplotlib.pyplot as plt

x =np.linspace(0, 10, 100)
z = np.linspace(-10, 0, 200)
X, Z, = meshgrid(x, z)
grb = 10
tht = .2
N2 = x*tht*np.sin(tht)*grb
M2 = x*tht*np.cos(tht)*grb



plt.figure()
#plt.plot(x, x*tht) 
#plt.plot(x, -x/tht+5) # reciprical slope
#plt.plot(x, *x + 0.1)
#plt.axis('equal')
#plt.ylim((0, 1))
N2 = 1
#M2 = 0.1
delta = np.tan(tht)**2
#delta = 1
M2 = N2*np.tan(tht)/delta

B = N2*Z + M2*X
plt.contour(x, z, B)
plt.plot(x, tht*x - 10)
plt.axis('equal')
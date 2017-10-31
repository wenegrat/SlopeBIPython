#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 09:34:18 2017

Testing out the Allen and Newberger boundary layer specification.
@author: jacob
"""
import numpy as np
import matplotlib.pyplot as plt

zhat = np.linspace(0, 1, 100)
Vo = .1
N2o = (3e-3)**2
Rzhat = zhat - 1
Rzhat2 = np.ones(zhat.shape)
tht = 5e-3
f = 1e-4
So = N2o*(1/f*np.tan(tht))**2

gamma  = 1
gamma = (1+So)**(-1)
V = Vo*(1 + Rzhat)
deltabbl = Vo*np.sin(tht)/(gamma*f*So)

Vz = Vo*(Rzhat2)/deltabbl
N2 = N2o*(1-gamma*Rzhat2)
Vx  = -f*So*gamma*Rzhat2
Ri = N2*f*(f+Vx)/(f*Vz)**2

delta = -So**(-1)*np.tan(tht)*Ri[0]**(1/2)
print('Ri: ' + str(Ri[0]))
print('delta:  ' + str(delta))
print('Vx/f:  ' + str(Vx[0]/f))
print('BL Thickness: ' + str(deltabbl))

plt.figure()
fig, ax = plt.subplots(1,3)
ax[0].plot(V, zhat)
ax[1].plot(N2, zhat)
ax[2].plot(Ri, zhat)

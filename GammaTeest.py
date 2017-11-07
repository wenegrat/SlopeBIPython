#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 14:52:19 2017

@author: jacob
"""
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import os

Vob = 0.1
No = 3.5e-3
f = 1e-4
theta = 5e-3
So = (No/f*np.tan(theta))**2

nz=1000
z = np.linspace(0, 1000, nz)

gammap = np.linspace(0.5, 1, 4)

Bz = No**2*np.ones(z.shape)
B = np.zeros(z.shape)
plt.figure()
Ri = np.zeros(gammap.shape)
counter = 0
for g in gammap:
    h = Vob*np.sin(theta)/(g*f*So)
#    h = Vob*f/(No**2*(1-g)*np.sin(theta))
    print(str(h))
    zind = np.floor( next((x[0] for x in enumerate(z) if x[1]>h)))
    Bz[0:zind] = No**2*(1-g)
    B[0:nz-1] = integrate.cumtrapz(Bz[::-1], z[::-1])[::-1]
    Ri[counter] = Bz[0]/(Vob/h)**2
    counter = counter+1
    plt.plot(Bz, z)

gammas = np.linspace(-2, -1.5, 4)

Bz = No**2*np.ones(z.shape)
B = np.zeros(z.shape)
plt.figure()
Rin = np.zeros(gammas.shape)
counter = 0
for g in gammas:
    h = -Vob*np.sin(theta)/(g*f*So)
#    h = Vob*f/  (No**2*(1-g)*np.sin(theta))

    print(str(h))
    zind = np.floor( next((x[0] for x in enumerate(z) if x[1]>h)))
    Bz[0:zind] = No**2*(1-g)
    B[0:nz-1] = integrate.cumtrapz(Bz[::-1], z[::-1])[::-1]
    Rin[counter] = Bz[0]/(Vob/h)**2
    counter = counter+1
    plt.plot(Bz, z)
#    plt.ylim((0,50))
#    plt.xlim((-0.0265, -0.024))
    
#%%
directoryname = "/home/jacob/Dropbox/Slope BI/EkmanUpFiles/"
directory = os.fsencode(directoryname)

directorynameout = "/home/jacob/dedalus/SlopeEkmanUp/"

ts = 10
filename = os.fsdecode(os.listdir(directory)[ts])
print(filename)
if filename.endswith(".npz"): 
    a = np.load(directoryname+filename);

plt.figure()
#plt.plot(a['b']+a['N']**2*a['z'], a['z'])
plt.plot(a['bz']+a['N']**2, a['z'])
btemp = a['N']**2*np.ones(a['z'].shape)
h = 70
gamma=-2.5
#h = -Vob*np.sin(theta)/(gamma*f*So)

zind = np.floor( next((x[0] for x in enumerate(a['z']) if x[1]>h)))
btemp[0:zind] = a['N']**2*(1-gamma*(h-a['z'][0:zind])/h)
plt.plot(btemp, a['z'])
plt.plot(Bz, z)
plt.ylim((0,100))

#%%
gamma = -1
h = 1000
plt.figure()
plt.plot(1-gamma*(h-z)/h, z)
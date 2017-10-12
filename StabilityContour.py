#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 15:45:36 2017

@author: jacob
"""
import sys
sys.path.append('/usr/share/')
import os
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 18})
directoryname = "./SlopeAngleRi100/"
directory = os.fsencode(directoryname)

ntht = 32
nll = 64

counter = 0
thetas = np.zeros(ntht)
gr = np.zeros((ntht, nll), dtype=np.float64)
plt.figure()
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    print(filename)
    if filename.endswith(".npz"): 
        a = np.load(directoryname+filename);
#        plt.semilogx(a['ll'], a['gr'])
        thetas[counter] = a['tht']
        gr[counter, :] = a['gr']#[:,-1]
        counter = counter + 1
        plt.plot(a['ll'], gr[counter-1,:]*np.sqrt(a['Bz'][-1])/(a['f']*a['Vz'][-1]))
        continue
    else:
        continue
idx = np.argsort(thetas)
thetas = thetas[idx]
gr = gr[idx,:]
grn =  (gr*np.sqrt(a['Bz'][-1])/(a['f']*a['Vz'][-1]))
grn = grn.astype(float)
plt.figure(figsize=(10, 6))
plt.grid(linestyle='--', alpha = 0.5)
plt.contourf(a['ll']*np.sqrt(a['Bz'][-1])*a['H']/a['f'], thetas*a['Bz'][-1]/(a['f']*a['Vz'][-1]), 
               grn,np.linspace(1e-2, .5, 50),vmin=-0.5, vmax=0.5, cmap='RdBu_r', labelsize=20)
#plt.contour(a['ll']*np.sqrt(a['Bz'][-1])*a['H']/a['f'], thetas*a['Bz'][-1]/(a['f']*a['Vz'][-1]), 
#               grn,np.linspace(1e-2, .5, 10),vmin=-0.5, vmax=0.5)
plt.colorbar()
plt.xlabel('$\hat{k}$', fontsize= 20)
plt.ylabel('$\delta$', fontsize=20)



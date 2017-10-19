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
directoryname = "../SlopeAngleRiVar/"
directory = os.fsencode(directoryname)

ntht = 64
nll = 64

counter = 0
rivec = np.zeros(ntht)
maxgr = np.zeros(ntht)
delta = np.zeros(ntht)
gr = np.zeros((ntht, nll), dtype=np.float64)    
grn = np.zeros((ntht, nll), dtype=np.float64)
plt.figure()
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    print(filename)
    if filename.endswith(".npz"): 
        a = np.load(directoryname+filename);
#        plt.semilogx(a['ll'], a['gr'])
        rivec[counter] = a['Ri']
        gr[counter, :] = a['gr']#[:,-1]
        maxgr[counter] = np.max(gr[counter,:])
        delta[counter] = a['Bz'][-1]/(a['f']*a['Vz'][-1])*a['tht']
        grn[counter,:] = (gr[counter,:]/a['f'])*(rivec[counter]**(1/2))
        counter = counter + 1
#        plt.plot(a['ll'], gr[counter-1,:]*np.sqrt(a['Bz'][-1])/(a['f']*a['Vz'][-1]))
        continue
    else:
        continue
idx = np.argsort(rivec)
rivec = rivec[idx]
maxgr = maxgr[idx]
delta = delta[idx]
gr = gr[idx,:]
grn = grn[idx,:]
grn = gr/a['f']
#grn =  (gr*np.sqrt(a['Bz'][-1])/(a['f']*a['Vz'][-1]))
#grn = gr/a['f']/(np.sqrt(a['Bz'][-1])/a['f']*a['tht'])
#S = np.sqrt(a['Bz'][-1])/a['f']*thetas
#grn = (gr*a['Bz'][-1]*a['f']/((a['f']*a['Vz'][-1])**2)*np.sqrt(np.sqrt(0.2*a['Bz'][-1]/a['f'])))
grn = grn.astype(float)
plt.figure(figsize=(10, 6))
plt.grid(linestyle='--', alpha = 0.5)
plt.contourf(a['ll']/(a['f']/(a['Vz'][-1]*a['H'])), rivec, 
               grn,np.linspace(1e-2, 0.25, 20),vmin=-0.2, vmax=0.2, cmap='RdBu_r', labelsize=20)
#plt.contour(a['ll']*np.sqrt(a['Bz'][-1])*a['H']/a['f'], thetas*a['Bz'][-1]/(a['f']*a['Vz'][-1]), 
#               grn,np.linspace(1e-2, .5, 10),vmin=-0.5, vmax=0.5)
plt.colorbar()
plt.xlabel('$\hat{l}$', fontsize= 20)
plt.ylabel('$Ri$', fontsize=20)

print("Maximum Ri processed: "+str(np.max(rivec[rivec!=0])))

plt.figure(figsize=(10, 6))
plt.plot(rivec, maxgr/a['f'])
#plt.ylim((-0.1, 0.1))


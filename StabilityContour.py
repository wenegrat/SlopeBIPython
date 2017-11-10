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
directoryname = "../SlopeAngleRi1/"
directory = os.fsencode(directoryname)

ntht = 64
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
        S = np.sqrt(a['Bz'][-1])/a['f']*a['tht']
#        gr[counter, :] = gr[counter,:]/np.abs(S)**(1/2)
        counter = counter + 1
        plt.plot(a['ll'], gr[counter-1,:]*np.sqrt(a['Bz'][-1])/(a['f']*a['Vz'][-1]))
        continue
    else:
        continue
idx = np.argsort(thetas)
thetas = thetas[idx]
gr = gr[idx,:]
grn =  (gr*np.sqrt(a['Bz'][-1])/(a['f']*a['Vz'][-1]))
#grn = grn/()
#grn = gr/a['f']
#S = np.sqrt(a['Bz'][-1])/a['f']*thetas

#grn = (gr*a['Bz'][-1]*a['f']/((a['f']*a['Vz'][-1])**2)*np.sqrt(np.sqrt(0.2*a['Bz'][-1]/a['f'])))
grn = grn.astype(float)

nc = 41
maxc = 0.5
fs =16

plt.figure(figsize=(10, 6))
plt.grid(linestyle='--', alpha = 0.5)
plt.contourf(a['ll']*np.sqrt(a['Bz'][-1])*a['H']/a['f'], thetas*a['Bz'][-1]/(a['f']*a['Vz'][-1]), 
               grn, np.linspace(0, maxc, nc),vmin=-maxc, vmax=maxc, cmap='RdBu_r', labelsize=20)
cbar = plt.colorbar()
cbar.set_ticks(np.linspace(0, 1, 11))
cbar.set_label('$\hat{\omega}$', fontsize=18)
CS = plt.contour(a['ll']*np.sqrt(a['Bz'][-1])*a['H']/a['f'], thetas*a['Bz'][-1]/(a['f']*a['Vz'][-1]), grn, 
            np.linspace(.1, 1, 10),colors='0.5' )
plt.tick_params(axis='both', which='major', labelsize=fs)
plt.clabel(CS, inline=1, fontsize = 10, fmt='%1.1f')
#plt.contour(a['ll']*np.sqrt(a['Bz'][-1])*a['H']/a['f'], thetas*a['Bz'][-1]/(a['f']*a['Vz'][-1]), 
#               grn,np.linspace(1e-2, .5, 10),vmin=-0.5, vmax=0.5)

plt.xlabel('$\hat{l}$', fontsize= 20)
plt.ylabel('$\delta$', fontsize=20)




print("Maximum \delta processed: "+str(np.max(thetas[thetas!=0])*a['Bz'][-1]/(a['f']*a['Vz'][-1])))


#plt.ylim((-0.1, 0.1))
#plt.savefig('/home/jacob/Dropbox/Slope BI/Slope BI Manuscript/StabilityRi1.eps', format='eps', dpi=1000)

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
import scipy.ndimage
from scipy.ndimage.filters import gaussian_filter
from scipy import interpolate
from pylab import *
plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 18})
directoryname = "../SlopeAngleRi1/"
directory = os.fsencode(directoryname)

ntht = 256
#ntht = 64
nll = 256


counter = 0
thetas = np.zeros(ntht)
gr = np.zeros((ntht, nll), dtype=np.float64)    
plt.figure()
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    print(filename)
    if filename.endswith(".npz"): 
        a = np.load(directoryname+filename);
        thetas[counter] = a['tht']
        gr[counter, :] = a['gr']#[:,-1]
        S = np.sqrt(a['Bz'][-1])/a['f']*a['tht']
        counter = counter + 1
        plt.plot(a['ll'], gr[counter-1,:]*np.sqrt(a['Bz'][-1])/(a['f']*a['Vz'][-1]))
        continue
    else:
        continue
idx = np.argsort(thetas)
thetas = thetas[idx]
gr = gr[idx,:]
grn = gr/a['f']

# DESPIKE
grnd = np.zeros(grn.shape)
grnd[1:-1,:] = grn[1:-1,:]-grn[0:-2,:]
grn[grnd>0.1] = np.nan
grnd[:, 1:-1] = grn[:, 1:-1]-grn[:,0:-2]
grn[np.abs(grnd)>0.05] = np.nan
array = np.ma.masked_invalid(grn)
ll, tt = np.meshgrid(a['ll'], thetas)
l1 = ll[~array.mask]
t1 = tt[~array.mask]
newg = array[~array.mask]
grn = interpolate.griddata((l1, t1), newg.ravel(), (ll, tt))
grn = grn.astype(float)

#%%
plt.rcParams['text.usetex'] = True
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

nc = 41
maxc = 0.5
#maxc = 0.2
fs =18

plt.figure(figsize=(6, 4))
plt.grid(linestyle='--', alpha = 0.5)
a0 = plt.contourf(a['ll']*np.sqrt(a['Bz'][-1])*a['H']/a['f'], thetas*a['Bz'][-1]/(a['f']*a['Vz'][-1]), 
               grn, np.linspace(0, maxc, nc),vmin=-maxc, vmax=maxc, cmap='RdBu_r', labelsize=20)
for c in a0.collections:
    c.set_edgecolor("face")
cbar = plt.colorbar(format='%1.2f')
#cbar.set_ticks(np.linspace(0, 1, 11))
cbar.set_ticks(np.linspace(0, maxc, 3))
cbar.set_label('Growth rate, ${\omega}_i/f$', fontsize=18)
#CS = plt.contour(a['ll']*np.sqrt(a['Bz'][-1])*a['H']/a['f'], thetas*a['Bz'][-1]/(a['f']*a['Vz'][-1]), grn, 
#            np.linspace(0.05, maxc, 4),colors='0.5' )
CS = plt.contour(a['ll']*np.sqrt(a['Bz'][-1])*a['H']/a['f'], thetas*a['Bz'][-1]/(a['f']*a['Vz'][-1]), grn, 
            np.linspace(0.1, maxc, 5),colors='0.5' )
plt.tick_params(axis='both', which='major', labelsize=fs)
plt.clabel(CS, inline=1, fontsize = 10, fmt='%1.2f')
#plt.contour(a['ll']*np.sqrt(a['Bz'][-1])*a['H']/a['f'], thetas*a['Bz'][-1]/(a['f']*a['Vz'][-1]), 
#               grn,np.linspace(1e-2, .5, 10),vmin=-0.5, vmax=0.5)
plt.xlim((0, 4))
plt.xlabel('Along-slope wavenumber, $l^*$', fontsize= 18)
plt.xlabel('Along-slope wavenumber, $(NH/f) l$', fontsize= 18)

plt.ylabel('Slope parameter, $\\alpha$', fontsize=18)
plt.yticks([-2, -1, 0, 1, 2])
plt.xticks([0, 1, 2, 3, 4])

plt.tight_layout()

print("Maximum \delta processed: "+str(np.max(thetas[thetas!=0])*a['Bz'][-1]/(a['f']*a['Vz'][-1])))

#plt.savefig('/home/jacob/Dropbox/Slope BI/Slope BI Manuscript/StabilityRi10.eps', format='eps', dpi=1000)
#plt.savefig('/home/jacob/Dropbox/Presentations/OS 2018 Slope Presentation/Working Files/Figures/StabilityRi1.pdf')


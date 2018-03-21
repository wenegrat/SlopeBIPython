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
directoryname = "../SlopeAngleRi10/"
directory = os.fsencode(directoryname)

ntht = 256
#ntht = 64
nll = 256


counter = 0
thetas = np.zeros(ntht)
gr = np.zeros((ntht, nll), dtype=np.float64)    
#plt.figure()
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    print(filename)
    if filename.endswith(".npz"): 
        a = np.load(directoryname+filename);
        thetas[counter] = a['tht']
        gr[counter, :] = a['gr']#[:,-1]
        S = np.sqrt(a['Bz'][-1])/a['f']*a['tht']
        counter = counter + 1
#        plt.plot(a['ll'], gr[counter-1,:]*np.sqrt(a['Bz'][-1])/(a['f']*a['Vz'][-1]))
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
grn[grnd>0.015] = np.nan
grnd[:, 1:-1] = grn[:, 1:-1]-grn[:,0:-2]
grn[np.abs(grnd)>0.015] = np.nan
array = np.ma.masked_invalid(grn)
ll, tt = np.meshgrid(a['ll'], thetas)
l1 = ll[~array.mask]
t1 = tt[~array.mask]
newg = array[~array.mask]
grn = interpolate.griddata((l1, t1), newg.ravel(), (ll, tt))
grn = grn.astype(float)

ll1 = a['ll']
bz1 = a['Bz']
h1 = a['H']
f1 = a['f']
vz1 = a['Vz']
grn1 = grn
thetas1=thetas

#%%
directoryname = "../SlopeAngleRi1/"
directory = os.fsencode(directoryname)

ntht = 256
#ntht = 64
nll = 256


counter = 0
thetas = np.zeros(ntht)
gr = np.zeros((ntht, nll), dtype=np.float64)    

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    print(filename)
    if filename.endswith(".npz"): 
        a = np.load(directoryname+filename);
        thetas[counter] = a['tht']
        gr[counter, :] = a['gr']#[:,-1]
        S = np.sqrt(a['Bz'][-1])/a['f']*a['tht']
        counter = counter + 1
#        plt.plot(a['ll'], gr[counter-1,:]*np.sqrt(a['Bz'][-1])/(a['f']*a['Vz'][-1]))
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
plt.rcParams.update({'font.size': 20})

nc = 41
maxc = 0.5
maxc1 = 0.16
fs =20

fig, ax=plt.subplots(1, 2, sharey=True, figsize=(12, 7))


ax[0].grid(linestyle='--', alpha = 0.5)
#a0 = plt.contourf(a['ll']*np.sqrt(a['Bz'][-1])*a['H']/a['f'], thetas*a['Bz'][-1]/(a['f']*a['Vz'][-1]), 
#               grn, np.linspace(0, maxc, nc),vmin=-maxc, vmax=maxc, cmap='RdBu_r', labelsize=20)
a0 =ax[0].contourf(ll1*np.sqrt(bz1[-1])*h1/f1, thetas1*bz1[-1]/(f1*vz1[-1]), 
               grn1, np.linspace(0, maxc1, nc),vmin=-maxc1, vmax=maxc1, cmap='RdBu_r', labelsize=20)
for c in a0.collections:
    c.set_edgecolor("face")
cbar = fig.colorbar(a0, ax=ax[0], format='%1.2f', orientation='horizontal')
#cbar.set_ticks(np.linspace(0, 1, 11))
cbar.set_ticks(np.linspace(0, maxc1, 3))
cbar.set_label('Growth rate, ${\omega}_i$', fontsize=fs)
cbar.solids.set_edgecolor("face")
#CS = plt.contour(a['ll']*np.sqrt(a['Bz'][-1])*a['H']/a['f'], thetas*a['Bz'][-1]/(a['f']*a['Vz'][-1]), grn, 
#            np.linspace(0.05, 0.2, 4),colors='0.5' )
#CS = plt.contour(a['ll']*np.sqrt(a['Bz'][-1])*a['H']/a['f'], thetas*a['Bz'][-1]/(a['f']*a['Vz'][-1]), grn, 
#            np.linspace(0.1, maxc, 5),colors='0.5' )
CS = ax[0].contour(ll1*np.sqrt(bz1[-1])*h1/f1, thetas1*bz1[-1]/(f1*vz1[-1]), grn1, 
            np.linspace(0.05, 0.2, 4),colors='0.5' )
ax[0].tick_params(axis='both', which='major', labelsize=fs)
plt.clabel(CS, inline=1, fontsize = 10, fmt='%1.2f')
#plt.contour(a['ll']*np.sqrt(a['Bz'][-1])*a['H']/a['f'], thetas*a['Bz'][-1]/(a['f']*a['Vz'][-1]), 
#               grn,np.linspace(1e-2, .5, 10),vmin=-0.5, vmax=0.5)
ax[0].set_xlim((0, 4))
ax[0].set_xlabel('Along-slope wavenumber, $l^*$', fontsize= fs)
#plt.xlabel('Along-slope wavenumber, $(NH/f) l$', fontsize= 18)

ax[0].set_ylabel('Slope parameter, $\\alpha$', fontsize=fs)
ax[0].set_yticks([-2, -1, 0, 1, 2])
ax[0].set_xticks([0, 1, 2, 3, 4])
bb = dict(boxstyle='Square', fc='w')
ax[0].text(0.15, 1.65, 'Ri$ = 10$', fontsize=20, bbox=bb)

##

ax[1].grid(linestyle='--', alpha = 0.5)
a0 = ax[1].contourf(a['ll']*np.sqrt(a['Bz'][-1])*a['H']/a['f'], thetas*a['Bz'][-1]/(a['f']*a['Vz'][-1]), 
               grn, np.linspace(0, maxc, nc),vmin=-maxc, vmax=maxc, cmap='RdBu_r', labelsize=20)
for c in a0.collections:
    c.set_edgecolor("face")
cbar = fig.colorbar(a0, ax=ax[1], format='%1.2f', orientation='horizontal')
#cbar.set_ticks(np.linspace(0, 1, 11))
cbar.set_ticks(np.linspace(0, maxc, 3))
cbar.set_label('Growth rate, ${\omega}_i$', fontsize=fs)
cbar.solids.set_edgecolor("face")
#CS = plt.contour(a['ll']*np.sqrt(a['Bz'][-1])*a['H']/a['f'], thetas*a['Bz'][-1]/(a['f']*a['Vz'][-1]), grn, 
#            np.linspace(0.05, 0.2, 4),colors='0.5' )
CS = ax[1].contour(a['ll']*np.sqrt(a['Bz'][-1])*a['H']/a['f'], thetas*a['Bz'][-1]/(a['f']*a['Vz'][-1]), grn, 
            np.linspace(0.1, maxc, 5),colors='0.5' )
#CS = ax[0].contour(ll1*np.sqrt(bz1[-1])*h1/f1, thetas1*bz1[-1]/(f1*vz1[-1]), grn1, 
#            np.linspace(0.05, 0.2, 4),colors='0.5' )
ax[1].tick_params(axis='both', which='major', labelsize=fs)
plt.clabel(CS, inline=1, fontsize = 10, fmt='%1.2f')
#plt.contour(a['ll']*np.sqrt(a['Bz'][-1])*a['H']/a['f'], thetas*a['Bz'][-1]/(a['f']*a['Vz'][-1]), 
#               grn,np.linspace(1e-2, .5, 10),vmin=-0.5, vmax=0.5)
ax[1].set_xlim((0, 4))
ax[1].set_xlabel('Along-slope wavenumber, $l^*$', fontsize= fs)
#plt.xlabel('Along-slope wavenumber, $(NH/f) l$', fontsize= 18)

#ax[0].set_ylabel('Slope parameter, $\\alpha$', fontsize=18)
ax[1].set_yticks([-2, -1, 0, 1, 2])
ax[1].set_xticks([0, 1, 2, 3, 4])

ax[1].text(0.15, 1.65, 'Ri$ = 1$', fontsize=20, bbox=bb)
plt.subplots_adjust(wspace=0.15, hspace=0)
#plt.tight_layout()


#plt.savefig('/home/jacob/Dropbox/Slope BI/Slope BI Manuscript/Revision 1/StabilityRiCombo.pdf', bbox_inches='tight')
#plt.savefig('/home/jacob/Dropbox/Presentations/OS 2018 Slope Presentation/Working Files/Figures/StabilityRi1.pdf')


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
from scipy import interpolate


plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 18})
directoryname = "../SlopeAngleRiVar2/"
directory = os.fsencode(directoryname)

ntht = 256
nll = 256

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
        delta[counter] = a['Bz'][-1]/(a['f']*a['Vz'][-1]*np.cos(a['tht']))*a['tht']
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

grnd = np.zeros(grn.shape)
grnd[1:-1,:] = grn[1:-1,:]-grn[0:-2,:]
grn[grnd>0.1] = np.nan
grnd[:, 1:-1] = grn[:, 1:-1]-grn[:,0:-2]
grn[np.abs(grnd)>0.05] = np.nan
#grn[grn==0] = np.nan
#valid_mask = ~np.isnan(grn)
#coords = np.array(np.nonzero(valid_mask)).T
#values = grn[valid_mask]
#it = interpolate.LinearNDInterpolator(coords, values)
#grn = it(list(np.ndindex(grn.shape))).reshape(grn.shape)

array = np.ma.masked_invalid(grn)
ll, tt = np.meshgrid(a['ll'], rivec)
l1 = ll[~array.mask]
t1 = tt[~array.mask]
newg = array[~array.mask]
grn = interpolate.griddata((l1, t1), newg.ravel(), (ll, tt))

#%%
nc = 41
maxc = 0.3
fs =18

grn = grn.astype(float)
plt.figure(figsize=(8, 4))
plt.grid(linestyle='--', alpha = 0.5)
ax = plt.contourf(a['ll']/(a['f']/(a['Vz'][-1]*a['H'])), rivec, 
               grn,np.linspace(0, maxc, nc),vmin=-maxc, vmax=maxc, cmap='RdBu_r', labelsize=20)
plt.xlim((0,3))
cbar = plt.colorbar()
cbar.set_ticks(np.linspace(0, maxc, 7))
cbar.set_label('${\omega}/f$', fontsize=18)
CS = plt.contour(a['ll']/(a['f']/(a['Vz'][-1]*a['H'])), rivec, grn, 
            np.linspace(.05, maxc, 6),colors='0.5' )
plt.tick_params(axis='both', which='major', labelsize=fs)
plt.clabel(CS, inline=1, fontsize = 10, fmt='%1.2f')
plt.xlabel('$l\'$', fontsize= 20)
plt.ylabel('$\mathrm{Ri}$', fontsize=20)
plt.xticks([0, 1, 2, 3])
plt.tight_layout()

print("Maximum Ri processed: "+str(np.max(rivec[rivec!=0])))
#plt.savefig('/home/jacob/Dropbox/Slope BI/Slope BI Manuscript/RiStability.eps', format='eps', dpi=1000)
#
#%%
## Make plot comparing Ri=1 with Stone solution
#k = a['ll']/(a['f']/(a['V'][-1]))
##k = a['ll']*np.sqrt(a['Bz'][-1])*a['H']/a['f']
#rind = 0
#Ri = rivec[rind]
#stgr = 1/(2*3**(1/2))*(k - 2/15*k**3*(Ri + 1))
#
#
#plt.figure()
#plt.plot(k, stgr)
#plt.plot(k, grn[rind,:])
#plt.plot(k, grtest/a['f'])
#plt.ylim((0, .5))
#plt.title(str(np.sqrt(a['Bz'][-1]*Ri)/a['f']*a['tht']))
#plt.legend(['Stone', '$S_H = 6\\times 10^{-2}$', '$S_H = 6\\times 10^{-3}$'])
##plt.figure(figsize=(10, 6))
##plt.plot(rivec, maxgr/a['f'])
#plt.ylim((-0.1, 0.1))

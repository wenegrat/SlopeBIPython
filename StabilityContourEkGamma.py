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

# LOAD BI
directoryname = "../EkmanGamma/"
directory = os.fsencode(directoryname)

ntht = 64
nll = 128

counter = 0
rivec = np.zeros(ntht)
maxgr = np.zeros(ntht)
delta = np.zeros(ntht)
gr = np.zeros((ntht, nll), dtype=np.float64)    
grn = np.zeros((ntht, nll), dtype=np.float64)
gam = np.zeros(ntht)
plt.figure()
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    print(filename)
    if filename.endswith(".npz"): 
        a = np.load(directoryname+filename);
#        plt.semilogx(a['ll'], a['gr'])

        gr[counter, :] = a['gr']#[:,-1]
        maxgr[counter] = np.max(gr[counter,:])
        delta[counter] = a['Bz'][0]/(a['f']*a['Vz'][0])*a['tht']
        gam[counter] = a['gamma']
        counter = counter + 1
#        plt.plot(a['ll'], gr[counter-1,:]*np.sqrt(a['Bz'][-1])/(a['f']*a['Vz'][-1]))
        continue
    else:
        continue
idx = np.argsort(gam)
gamBI = gam[idx]
maxgrBI = maxgr[idx]
deltaBI = delta[idx]
grBI = gr[idx,:]
grnBI = grn[idx,:]
llBI = a['ll']
So = (5e-3)**2/a['f']**2*a['tht']**2
grnBI = grBI/(a['f']*So) # Normalized by spin-down timescale


#SI
directoryname = "../EkmanGammaSI/"
directory = os.fsencode(directoryname)

ntht = 32
nll = 128

counter = 0
rivec = np.zeros(ntht)
maxgr = np.zeros(ntht)
delta = np.zeros(ntht)
gr = np.zeros((ntht, nll), dtype=np.float64)    
grn = np.zeros((ntht, nll), dtype=np.float64)
gam = np.zeros(ntht)
plt.figure()
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    print(filename)
    if filename.endswith(".npz"): 
        a = np.load(directoryname+filename);
#        plt.semilogx(a['ll'], a['gr'])

        gr[counter, :] = a['gr']#[:,-1]
        maxgr[counter] = np.max(gr[counter,:])
        delta[counter] = a['Bz'][0]/(a['f']*a['Vz'][0])*a['tht']
        gam[counter] = a['gamma']
        counter = counter + 1
#        plt.plot(a['ll'], gr[counter-1,:]*np.sqrt(a['Bz'][-1])/(a['f']*a['Vz'][-1]))
        continue
    else:
        continue
idx = np.argsort(gam)
gam = gam[idx]
maxgr = maxgr[idx]
delta = delta[idx]
gr = gr[idx,:]
#grn = grn[idx,:]
grn = gr/(a['f']*So) # Normalized by spin-down timescale

#grn =  (gr*np.sqrt(a['Bz'][-1])/(a['f']*a['Vz'][-1]))
#grn = gr/a['f']/(np.sqrt(a['Bz'][-1])/a['f']*a['tht'])
#S = np.sqrt(a['Bz'][-1])/a['f']*thetas
#grn = (gr*a['Bz'][-1]*a['f']/((a['f']*a['Vz'][-1])**2)*np.sqrt(np.sqrt(0.2*a['Bz'][-1]/a['f'])))

#%%
nc = 41
maxc = 6#0.3
fs =18

ln = a['ll']*np.abs(a['V'][-1]/a['f'])
lnBI = llBI*np.abs(a['V'][-1]/a['f'])
#ln = a['ll']
grn = grn.astype(float)

fig, ax = plt.subplots(1,2, sharey=True, figsize=(10, 5.5))
IM1 = ax[0].contourf(lnBI, gamBI, grnBI, np.linspace(0, maxc, nc),vmin=-maxc, vmax=maxc, cmap='RdBu_r', labelsize=20)
CS = ax[0].contour(lnBI, gamBI, grnBI, list(range(1, 10, 1)), colors='0.5')
plt.clabel(CS, inline=1, fontsize = 10, fmt='%1.2f')
ax[0].set_ylim((0.5, 1))
ax[0].set_xlim((0, 3))
ax[0].plot(ln, (1+So)**(-1)*np.ones(a['ll'].shape), color='k')
ax[0].set_xlabel('$\hat{l}$', fontsize=fs)
ax[0].set_ylabel('$\gamma$', fontsize=fs)
cbar1 = fig.colorbar(IM1, orientation='horizontal', ax=ax[0])
cbar1.set_ticks(np.linspace(0, maxc, 3))
cbar1.set_label('$\hat{\omega}$', fontsize=18)

maxc = 24
IM = ax[1].contourf(ln, gam, grn,np.linspace(0, maxc, nc),vmin=-maxc, vmax=maxc, cmap='RdBu_r', labelsize=20)
CS = ax[1].contour(ln, gam, grn, list(range(1, 50, 5)), colors='0.5')
plt.clabel(CS, inline=1, fontsize = 10, fmt='%1.2f')
ax[1].plot(ln, (1+So)**(-1)*np.ones(a['ll'].shape), color='k')
ax[1].set_xlabel('$\hat{k}$', fontsize=fs)
plt.tight_layout()
ax[1].set_xlim((0, 10))
cbar = plt.colorbar(IM, orientation='horizontal')
cbar.set_ticks(np.linspace(0, maxc, 3))
cbar.set_label('$\hat{\omega}$', fontsize=18)
#cbar = fig.colorbar(IM,  ax=ax.ravel().tolist(), orientation='horizontal', shrink = 0.5)
#cbar.set_label('$\hat{\omega}$', fontsize=18)
#cbar.set_ticks(np.linspace(0, maxc, 3))


#plt.savefig('/home/jacob/Dropbox/Slope BI/Slope BI Manuscript/EkGamma.eps', format='eps', dpi=1000)


#%%
#plt.figure(figsize=(10, 6))
#plt.grid(linestyle='--', alpha = 0.5)
#ax = plt.contourf(ln, gam, 
#               grn,np.linspace(0, maxc, nc),vmin=-maxc, vmax=maxc, cmap='RdBu_r', labelsize=20)
##plt.ylim((0.3,1))
#cbar = plt.colorbar()
##cbar.set_ticks(np.linspace(0, 1, 11))
#cbar.set_label('$\hat{\omega}$', fontsize=18)
#CS = plt.contour(ln, gam, grn, 
#            np.linspace(1, maxc, 10),colors='0.5' )
#plt.plot(ln, (1+So)**(-1)*np.ones(a['ll'].shape), color='k')
#plt.tick_params(axis='both', which='major', labelsize=fs)
#plt.clabel(CS, inline=1, fontsize = 10, fmt='%1.2f')
#plt.xlabel('$\hat{l}$', fontsize= 20)
#plt.ylabel('$\gamma$', fontsize=20)
##plt.xlim((0, 3))
#print("Maximum Gamma processed: "+str(np.max(gam[gam!=0])))

#plt.figure(figsize=(10, 6))
#plt.plot(rivec, maxgr/a['f'])
#plt.ylim((-0.1, 0.1))
#plt.savefig('/home/jacob/Dropbox/Slope BI/Slope BI Manuscript/EkGamma.eps', format='eps', dpi=1000)


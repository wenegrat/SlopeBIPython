#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make plots for varying theta angle.
Created on Mon Oct  9 11:43:31 2017

@author: jacob
"""
import os
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from pylab import *

nll = 32

directoryname = "../GOTMEkman/"
directory = os.fsencode(directoryname)
plt.figure
counter = 0
thetas = list(np.zeros(np.array(os.listdir(directory)).shape))
grt = (np.zeros(np.array(os.listdir(directory)).shape))
time = (np.zeros(np.array(os.listdir(directory)).shape))
wave = (np.zeros(np.array(os.listdir(directory)).shape))
trans = (np.zeros(np.array(os.listdir(directory)).shape))

for file in sorted(os.listdir(directory)):
    filename = os.fsdecode(file)
    print(filename)
    if filename.endswith(".npz"): 
        a = np.load(directoryname+filename);
        trans[counter] = integrate.trapz(a['U'], a['z']) # Cross-slope tranpsort

            
        
        if a['gr'][0]/a['f']<0.05:
            plt.plot(a['ll'], a['gr']/a['f'])

            plt.xlabel('along-track wavenumber [m$^{-1}$]')
            plt.ylabel('growth rate')
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
#        plt.ylim((0,1))
            plt.tight_layout()
            
            thetas[counter] = str(np.real(np.tanh(a['tht'])*a['Bz'][-1]/(a['f']*a['Vz'][-1])))
            grt[counter] = max(a['gr'])
            time[counter] = a['time'].item(0)
            ind = np.argsort(a['gr'])
            wave[counter] = a['ll'][ind[-1]]
        counter = counter + 1
        continue
    else:
        continue
idx = np.argsort(time)
grt= grt[idx]
time= time[idx]
trans = trans[idx]
dtransdt = np.gradient(trans, 3600*12)
ttheory = -trans/dtransdt
#wave = wave[idx]**(-1)*a['N']*a['H']/a['f']
#plt.legend(thetas)
        #%%%
plt.figure()
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(12, 6))
ax[0].plot(time*a['f']/(2*np.pi), grt/a['f'], marker="x", linestyle='None')
ax[0].plot(time*a['f']/(2*np.pi), 2*np.pi/(ttheory)*1/a['f'], marker="+", linestyle='None')
#ax[0].set_ylim(0, 0.3)
ax[0].set_ylabel('$\omega$/f')
#ax[0].set_ylim((-2,2))
ax[0].grid()
#    
ax[1].plot(time*a['f']/(2*np.pi), trans)
#ax[1].plot(time*a['f']/(2*np.pi), wave, marker="x", linestyle='None')
#ax[1].set_ylim(0, 1e5)
#ax[1].set_ylabel('$l/(f/NH)$')
#ax[1].set_xlabel('$t/(1/f)$')
#ax[1].grid()

#%%
z = a['z']    

# Make Background State Plot
fig, ax = plt.subplots(1, 3, sharey=True)

ax[0].plot(a['kap'], z)
#ax[]
#ax[0].set_xlabel('mixing coefficient [m$^2$/s]', va='baseline')
ax[0].set_ylabel('slope-normal coordinate [m]')
#ax[0].get_xaxis().set_label_coords(.5, -.12)

V = np.array([a['V']])
B = np.array([a['B']])
Vz = np.gradient(V[-1,:], np.gradient(z))
Bz = np.gradient(B[-1,:], np.gradient(z))
Ri = Bz/(Vz**2)
ax[0].plot(a['nu'], z)
ax[0].set_ylabel('slope-normal coordinate [m]')

ax[1].plot(a['U'], z)
ax[1].plot(a['V'], z)
ax[1].set_xlabel('mean flow [m/s]', va='baseline')
ax[1].get_xaxis().set_label_coords(.5, -.12)

ax[2].plot(a['B'], z)
ax[2].set_xlabel('mean buoyancy [m/s$^2$]', va='baseline')
ax[2].get_xaxis().set_label_coords(.5, -.12)

#%%
z = a['z']    

plt.figure()
plt.plot(a['b']*a['w'], z)
#
## Make Energetics Plot
#plt.figure(figsize=(4.8, 4.8))
#plt.plot(a['SP'], z)
#plt.plot(a['BP'], z)
#plt.xlabel('kinetic energy tendency [m$^2$/s$^3$]')
#plt.ylabel('slope-normal coordinate [m]')
#plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
#plt.legend(['shear production', 'buoyancy production'], frameon=False)
#plt.tight_layout()
#
## Make Structure Plot
#ly = np.linspace(0, 2*np.pi, a['nz'])
#
#fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(6.4, 6.4))
#im = ax[0,0].pcolormesh(ly, z, np.real(a['u'].reshape(a['nz'], 1)
#        * np.exp(1j*ly.reshape(1,a['nz']))), rasterized=True, cmap='RdBu_r')
#plt.colorbar(im, ax=ax[0,0])
#ax[0,0].set_title('across-slope velocity')
#im = ax[0,1].pcolormesh(ly, z, np.real(a['v'].reshape(a['nz'], 1)
#        * np.exp(1j*ly.reshape(1,a['nz']))), rasterized=True, cmap='RdBu_r')
#plt.colorbar(im, ax=ax[0,1])
#ax[0,1].set_title('along-slope velocity')
#im = ax[1,0].pcolormesh(ly, z, np.real(a['w'].reshape(a['nz'], 1)
#        * np.exp(1j*ly.reshape(1,a['nz']))), rasterized=True, cmap='RdBu_r')
#plt.colorbar(im, ax=ax[1,0])
#ax[1,0].set_title('slope-normal velocity')
#im = ax[1,1].pcolormesh(ly, z, np.real(a['b'].reshape(a['nz'], 1)
#        * np.exp(1j*ly.reshape(1,a['nz']))), rasterized=True, cmap='RdBu_r')
#plt.colorbar(im, ax=ax[1,1])
#ax[1,1].set_title('buoyancy')
#ax[0,0].set_xticks([0, np.pi, 2*np.pi])
#ax[1,0].set_xlabel('phase')
#ax[1,1].set_xlabel('phase')
#ax[0,0].set_ylabel('slope-normal coordinate [m]')
#ax[1,0].set_ylabel('slope-normal coordinate [m]')
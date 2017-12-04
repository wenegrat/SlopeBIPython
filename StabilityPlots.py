#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make plots for varying theta angle.
Created on Mon Oct  9 11:43:31 2017

@author: jacob
"""
import os
import numpy as np
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
from pylab import *

directoryname = "../DeepShallowModes/"
directory = os.fsencode(directoryname)

ntht = 5
nll = 256

plt.figure(figsize=(4.8, 4.8))
counter = 0
thetas = np.zeros(np.array(os.listdir(directory)).shape)
gr = np.zeros((ntht, nll), dtype=np.float64)    
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    print(filename)
    if filename.endswith(".npz"): 
        a = np.load(directoryname+filename);
#        plt.semilogx(a['ll'], a['gr'])
        plt.plot(a['ll'], a['gr']/(a['f']))
        plt.xlabel('along-track wavenumber [m$^{-1}$]')
        plt.ylabel('growth rate')
        plt.ylim((0, .5))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.tight_layout()        
        thetas[counter] = (a['tht'])
        gr[counter,:] = a['gr']
        counter = counter + 1
        continue
    else:
        continue
#plt.legend(thetas)
idx = np.argsort(thetas)

thetas = thetas[idx]
gr = gr[idx,:]
grn = gr/a['f']
#%%
plt.rcParams['text.usetex'] = True

def tickfun(X):
    Y = 1/X/1000
    return ['%i' % z for z in Y]



#fig, ax = plt.subplots(1, 2, sharey=False, figsize=(12, 5))
fs = 20
plt.rc('xtick', labelsize=fs)
plt.rc('ytick', labelsize=fs)

fig = plt.figure(figsize=(12,5))
ax1 = fig.add_subplot(111)
#ax2 = ax1.twiny()
for i in range(0, thetas.shape[0]):
    
    ax1.semilogx(a['ll']/(2*np.pi), grn[i,:], label='$\\alpha = $ ' + str(thetas[i]), linewidth=2)
    ax1.set_xlabel('Along-slope wavenumber [m$^{-1}$]', fontsize = fs)
    ax1.set_ylabel('Growth rate', fontsize=fs)
    ax1.set_ylim((0, .25))
    ax1.set_xlim((1e-4, 1e-2))
    ax1.set_xlim((2e-5, 1.5e-3))
    plt.grid(linestyle='--', alpha = 0.5)

#    ax[0].semilogx(a['ll'], grn[i,:])
#    ax[0].set_xlabel('along-track wavenumber [m$^{-1}$]')
#    ax[0].set_ylabel('growth rate')
#    ax[0].set_ylim((0, .23))
#    ax[0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
#    
#    ax[1].plot(a['ll'], grn[i,:])
#    ax[1].set_xlabel('along-track wavenumber [m$^{-1}$]')
#    ax[1].set_ylabel('growth rate')
#    ax[1].set_ylim((0, .04))
#    ax[1].set_xlim((0, 0.0003))
#    ax[1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax1.legend(fontsize=fs)
ax1.grid(linestyle='--', alpha = 0.5, which='Both')

newticks = np.array([2*np.pi/50e3, 2*np.pi/10e3, 2*np.pi/1e3])
newticks = np.array([1/50e3, 1/10e3, 1e-3])
#ax2.set_xscale('log')
#
#ax2.set_xticks(newticks)
#ax2.set_xlim(ax1.get_xlim())
#
#ax2.set_xticklabels(tickfun(newticks))
#ax2.set_xlabel('Wavelength [km]', labelpad=10, fontsize=fs)
#ax2.grid(False)
z = a['z']    

#plt.savefig('/home/jacob/Dropbox/Slope BI/Slope BI Manuscript/IdealizedGrowthRates.eps', format='eps', dpi=1000, bbox_inches='tight')

#%%
# Make Background State Plot

V = np.array([a['V']])
B = np.array([a['B']])
Vz = np.gradient(V[-1,:], np.gradient(z))
Bz = np.gradient(B[-1,:], np.gradient(z))
Ri = Bz/(Vz**2)
plt.figure(figsize=(2, 4))
plt.semilogx(Ri, z)
plt.xticks([1, 1e1, 1e2])
plt.ylabel('Height Above Bottom [m]', fontsize=fs)
plt.xlabel('Ri', fontsize=fs)
plt.grid(linestyle='--', alpha = 0.5)
plt.ylim((0, 1000))

#plt.savefig('/home/jacob/Dropbox/Slope BI/Slope BI Manuscript/RiStructure.eps', format='eps', dpi=1000, bbox_inches='tight')

#%%
# Make Background State Plot

V = np.array([a['V']])
B = np.array([a['B']])
Vz = np.gradient(V[-1,:], np.gradient(z))
Bz = np.gradient(B[-1,:], np.gradient(z))
Ri = Bz/(Vz**2)
fig, ax = plt.subplots(1,2, sharey=True, figsize=(6,6))

ax[0].plot(a['V'], z)
ax[0].set_xticks([-0., 5e-2, 0.1])
ax[0].set_xlim((-0.01, .11))
ax[0].set_xlabel('V ($ms^{-1}$)', fontsize=fs, color='b')
ax2 = ax[0].twiny()
ax2.plot(a['B'], z, color='g')
ax2.set_xticks([0, 5e-4, 0.001])
ax2.set_xlim((0-1e-4, 0.0011))
for tl in ax2.get_xticklabels():
    tl.set_color('g')
for tl in ax[0].get_xticklabels():
    tl.set_color('b')
ax[0].set_ylabel('Height Above Bottom [m]', fontsize=fs)
ax2.grid(linestyle='--', alpha=0.5)
ax[0].grid(linestyle='--', alpha=0.5)
ax2.set_xlabel('B ($m s^{-2}$)', fontsize=fs, color='g')
ax[1].semilogx(Ri, z)
ax[1].set_xticks([1, 1e1, 1e2])
ax[1].set_xlabel('Ri', fontsize=fs)
ax[1].grid(linestyle='--', alpha = 0.5)
ax[1].set_ylim((0, 1000))

#plt.savefig('/home/jacob/Dropbox/Slope BI/Slope BI Manuscript/RiStructure.eps', format='eps', dpi=1000, bbox_inches='tight')

#%%
#fig, ax = plt.subplots(1, 3, sharey=True, figsize=(12, 7))
#
#ax[0].semilogx(a['kap'], z)
#ax[0].set_xlabel('mixing coefficient [m$^2$/s]', va='baseline')
#ax[0].set_ylabel('Height Above Bottom [m]')
#ax[0].get_xaxis().set_label_coords(.5, -.12)
#
#V = np.array([a['V']])
#B = np.array([a['B']])
#Vz = np.gradient(V[-1,:], np.gradient(z))
#Bz = np.gradient(B[-1,:], np.gradient(z))
#Ri = Bz/(Vz**2)
#ax[0].plot(Ri, z)
#ax[0].set_ylabel('slope-normal coordinate [m]')
#
#ax[1].plot(a['U'], z)
#ax[1].plot(a['V'], z)
#ax[1].set_xlabel('mean flow [m/s]', va='baseline')
#ax[1].get_xaxis().set_label_coords(.5, -.12)
#
#ax[2].plot(a['B'], z)
#ax[2].set_xlabel('mean buoyancy [m/s$^2$]', va='baseline')
#ax[2].get_xaxis().set_label_coords(.5, -.12)

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
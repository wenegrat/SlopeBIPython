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
from scipy import interpolate
from scipy.ndimage.filters import gaussian_filter

import matplotlib.pyplot as plt
plt.rc('text', usetex=True)

from pylab import *
import glob

nll = 192
nz = 256
#nz = 128
directoryname = "../GOTMEkmanCluster/"
directoryname = "../GOTMEkmanU/"
directory = os.fsencode(directoryname)
plt.figure
counter = 0
thetas = list(np.zeros(np.array(os.listdir(directory)).shape))
grt = (np.zeros(np.array(os.listdir(directory)).shape))
time = (np.zeros(np.array(os.listdir(directory)).shape))
wave = (np.zeros(np.array(os.listdir(directory)).shape))
trans = (np.zeros(np.array(os.listdir(directory)).shape))

def keyfunc(x):
    junk, nend = x.split('_')
    num, dot, ty = nend.partition('.')
    return int(num)

listd = os.listdir(directory)

dirfiles = sorted(glob.glob(directoryname+'*.npz'), key=keyfunc)
#dirfiles = dirfiles[0:5]
#dirfiles.sort(key=lambda f:int(''.join(filter(str.isdigit, f))) )
grf = np.nan*np.zeros((len(dirfiles), nll))
#time = np.nan*np.zeros((len(dirfiles)))
Us = np.nan*np.zeros((len(dirfiles), nz))
Us = np.nan*np.zeros((len(dirfiles), nz))
Vs = np.nan*np.zeros((len(dirfiles), nz))
Nus = np.nan*np.zeros((len(dirfiles), nz))
Kappas = np.nan*np.zeros((len(dirfiles), nz))
Bs = np.nan*np.zeros((len(dirfiles), nz))
Bzf = np.nan*np.zeros((len(dirfiles), nz))
for filename in dirfiles:
#    filename = os.fsdecode(file)
    print(filename)
    if filename.endswith(".npz"): 
        a = np.load(directoryname+filename);
        trans[counter] = integrate.trapz(a['U'][0:50], a['z'][0:50]) # Cross-slope tranpsort

        time[counter] = a['time'].item(0) 
        
#        if counter == 10: # use to look at particular energetics.
#            break
        ke = integrate.trapz(a['SP']+a['BP']+a['DISS'], a['z'])
        if a['gr'][0]/a['f']<.025*1000:
#        if ke>0:
            plt.plot(a['ll'], a['gr']/a['f'])

            plt.xlabel('along-track wavenumber [m$^{-1}$]')
            plt.ylabel('growth rate')
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
#        plt.ylim((0,1))
            plt.tight_layout()
            
            thetas[counter] = str(np.real(np.tanh(a['tht'])*a['Bz'][-1]/(a['f']*a['Vz'][-1])))
            grt[counter] = max(a['gr'])

            ind = np.argsort(a['gr'])
            wave[counter] = a['ll'][ind[-1]]
            grf[counter,:] = a['gr']
            Us[counter,:]  = a['U']
            Vs[counter,:]  = a['V']
            Kappas[counter,:]  = a['kap']
            Nus[counter,:]  = a['nu']   
            Bs[counter,:]  = a['B']            
            Bzf[counter,:] = a['Bz']
        counter = counter + 1
        continue
    else:
        continue
#idx = np.argsort(time)
#grt= grt[idx]
#time= time[idx]
#trans = trans[idx]
dtransdt = np.gradient(trans, 3600*12)
ttheory = -trans/dtransdt
#wave = wave[idx]**(-1)*a['N']*a['H']/a['f']
#plt.legend(thetas)
#%% Calculate and Plot Delta
bs = (3.5e-3)**2
M2 = -(Bzf - bs)*np.sin(tht)
N2 = (Bzf - bs)*np.cos(tht) + bs
N2hat = Bzf
M2hat = bs*np.sin(tht)*np.ones(Bzf.shape)
delta = np.abs(N2/M2)*np.tan(tht)
ang = np.arctan(N2/M2)
angr = np.arctan(N2hat/M2hat) + tht
plt.figure()
#plt.plot(time, np.log10(delta[:, 0]))
#plt.pcolor(time, z, np.transpose(delta))
#plt.colorbar()
#plt.ylim((-3, 0))
#plt.xlim((-1,1000))
plt.plot(np.pi/2+angr[:,0])
plt.plot(np.pi/2 + ang[:,0])
        #%%%
plt.figure()
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(12, 6))
ax[0].plot(time*a['f']/(2*np.pi), grt/a['f'], marker="x", linestyle='None')
ax[0].plot(time*a['f']/(2*np.pi), 2*np.pi/(ttheory)*1/a['f'], marker="+", linestyle='None')
#ax[0].set_ylim(0, 0.3)
ax[0].set_ylabel('$\omega$/f')
ax[0].set_ylim((0,0.5))
ax[0].grid()
#    
ax[1].plot(time*a['f']/(2*np.pi), trans)
#ax[1].plot(time*a['f']/(2*np.pi), wave, marker="x", linestyle='None')
#ax[1].set_ylim(0, 1e5)
#ax[1].set_ylabel('$l/(f/NH)$')
#ax[1].set_xlabel('$t/(1/f)$')
#ax[1].grid()
#%%
plt.rcParams.update({'font.size': 18})
plt.rcParams['contour.negative_linestyle'] = 'solid'
cm = 'seismic'
z = a['z'] 
um = 0.025
nc = 24#30
uc = np.linspace(-um, um,nc)
vm = 0.11
vc = np.linspace(-vm, vm, nc)
bm = 0.0065 
#bm = 0.0075
bma = 0.0045 
bc = np.linspace(-bm, -bma, nc)
kc = np.linspace(-5, -2, 10)
fig, ax = plt.subplots( 5, 1, sharex=True, figsize=(10, 10))
UM = ax[0].contourf(time*a['f']/(2*np.pi), z, np.transpose(Us),uc, vmin=-um, vmax=um, cmap=cm )
cb = fig.colorbar(UM, ax=ax[0])
cb.set_ticks(np.linspace(uc[0], uc[-1], 3))
cb.set_label('$m s^{-1}$')

ax[0].contour(time*a['f']/(2*np.pi), z, np.transpose(Us),levels=uc, colors='0.5')
VM = ax[1].contourf(time*a['f']/(2*np.pi), z, np.transpose(Vs),vc, vmin=-vm, vmax=vm, cmap=cm, extend='both' )
VM.set_clim(-0.1, 0.1)
cb = fig.colorbar(VM, ax=ax[1])
cb.set_ticks(np.linspace(-.1,.1, 3))
cb.set_label('$m s^{-1}$')

ax[1].contour(time*a['f']/(2*np.pi), z, np.transpose(Vs),levels=vc, colors='0.5')

BM = ax[2].contourf(time*a['f']/(2*np.pi), z, np.transpose(Bs), bc, vmin=-bm, bmax=-bma, cmap=cm )
cb = fig.colorbar(BM, ax=ax[2])
cb.set_ticks(np.linspace(bc[0], bc[-1], 3))
cb.set_label('$m s^{-2}$')

ax[2].contour(time*a['f']/(2*np.pi), z, np.transpose(Bs),levels=bc, colors='0.5')

KM = ax[3].contourf(time*a['f']/(2*np.pi), z, np.log10(np.transpose(Kappas)), kc, vmin=kc[0], vmax=kc[-1], cmap=cm)
cb = fig.colorbar(KM, ax=ax[3])
cb.set_ticks(np.linspace(kc[0], kc[-1], 3))
cb.set_label('$log_{10}(m^2 s^{-1})$')

ax[3].contour(time*a['f']/(2*np.pi), z, np.log10(np.transpose(Kappas)),levels=kc, colors='0.5')

NM = ax[4].contourf(time*a['f']/(2*np.pi), z, np.log10(np.transpose(Nus)), kc, vmin=kc[0], vmax=kc[-1], cmap=cm)
cb = fig.colorbar(NM, ax=ax[4])
cb.set_ticks(np.linspace(kc[0], kc[-1], 3))
cb.set_label('$log_{10}(m^2 s^{-1})$')
ax[4].contour(time*a['f']/(2*np.pi), z, np.log10(np.transpose(Nus)),levels=kc, colors='0.5')

maxh = 100
ax[0].set_ylim((0, maxh))
ax[1].set_ylim((0, maxh))
ax[2].set_ylim((0, maxh))
ax[3].set_ylim((0, maxh))
ax[4].set_ylim((0, maxh))
maxt = 30
ax[0].set_xlim((0, maxt))
ax[1].set_xlim((0, maxt))
ax[2].set_xlim((0, maxt))
ax[3].set_xlim((0, maxt))
ax[4].set_xlim((0, maxt))
ax[0].set_ylabel('$z (m)$', fontsize=26)
ax[1].set_ylabel('$z (m)$', fontsize=26)
ax[2].set_ylabel('$z (m)$', fontsize=26)
ax[3].set_ylabel('$z (m)$', fontsize=26)
ax[4].set_ylabel('$z (m)$', fontsize=26)
ax[4].set_xlabel('$\\frac{tf}{2 \\pi}$', fontsize=26)

bb = dict(boxstyle='Square', fc='w')

xp = 0.75
ax[0].text(xp, 75, '$U$', fontsize=20, bbox=bb)
ax[1].text(xp, 75, '$V$', fontsize=20, bbox=bb)
ax[2].text(xp, 75, '$B$', fontsize=20, bbox=bb)
ax[3].text(xp, 75, '$\\kappa$', fontsize=20, bbox=bb)
ax[4].text(xp, 75, '$\\nu$', fontsize=20, bbox=bb)

#plt.savefig('/home/jacob/Dropbox/Slope BI/Slope BI Manuscript/EkmanSpinDOWN.eps', format='eps', dpi=1000)
#plt.savefig('/home/jacob/Dropbox/Slope BI/Slope BI Manuscript/EkmanSpinUP.eps', format='eps', dpi=1000)

#%%
#plt.figure()
#plt.contourf(time, z, np.transpose(Nus))
#plt.colorbar()
#%%
plt.figure()
plt.plot(Nus[-1,1:-1]-Nus[-1, 0:-2], z[1:-1])
box = np.ones(8)/8
Ns = np.convolve(Nus[-1,:], box, mode='same')
plt.plot(Ns[1:-1]-Ns[0:-2], z[1:-1])
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
plt.figure()
plt.plot(Ri, z)
plt.ylim((0,100))
plt.xlim((-1, 1))

#%%
z = a['z']    

#plt.figure()
#plt.plot(a['b']*a['w'], z)
#
## Make Energetics Plot
plt.figure(figsize=(4.8, 4.8))
plt.plot(a['SP'], z)
plt.plot(a['BP'], z)
plt.plot(a['DISS'], z)
plt.xlabel('kinetic energy tendency [m$^2$/s$^3$]')
plt.ylabel('slope-normal coordinate [m]')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
plt.legend(['shear production', 'buoyancy production', 'Dissipation'], frameon=False)
plt.tight_layout()
plt.ylim((0, 100))
#
## Make Structure Plot
#ly = np.linspace(0, 2*np.pi, a['nz'])
#
nz = z.shape[0]
ly = np.linspace(0, 2*np.pi, nz)
fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(6.4, 6.4))
im = ax[0,0].pcolormesh(ly, z, np.real(a['u'].reshape(a['nz'], 1)
        * np.exp(1j*ly.reshape(1,a['nz']))), rasterized=True, cmap='RdBu_r')
plt.colorbar(im, ax=ax[0,0])
ax[0,0].set_title('across-slope velocity')
im = ax[0,1].pcolormesh(ly, z, np.real(a['v'].reshape(a['nz'], 1)
        * np.exp(1j*ly.reshape(1,a['nz']))), rasterized=True, cmap='RdBu_r')
plt.colorbar(im, ax=ax[0,1])
ax[0,1].set_title('along-slope velocity')
im = ax[1,0].pcolormesh(ly, z, np.real(a['w'].reshape(a['nz'], 1)
        * np.exp(1j*ly.reshape(1,a['nz']))), rasterized=True, cmap='RdBu_r')
plt.colorbar(im, ax=ax[1,0])
ax[1,0].set_title('slope-normal velocity')
im = ax[1,1].pcolormesh(ly, z, np.real(a['b'].reshape(a['nz'], 1)
        * np.exp(1j*ly.reshape(1,a['nz']))), rasterized=True, cmap='RdBu_r')
plt.colorbar(im, ax=ax[1,1])
ax[1,1].set_title('buoyancy')
ax[0,0].set_xticks([0, np.pi, 2*np.pi])
ax[1,0].set_xlabel('phase')
ax[1,1].set_xlabel('phase')
ax[0,0].set_ylabel('slope-normal coordinate [m]')
ax[1,0].set_ylabel('slope-normal coordinate [m]')
ax[0,0].set_ylim((0,100))
#%%
#timet = time
#timet[timet==0] = np.nan
#grf[grf == 0] = np.nan

#grnd = np.zeros(grf.shape)
#grnd[1:-1, :] = grf[1:-1,:]-grf[ 0:-2, :]
#grnd[np.isnan(grnd)] =0
#grf[np.abs(grnd)>1e-7] = np.nan

array = grf
x = np.arange(0, array.shape[1])
y = np.arange(0, array.shape[0])
array = np.ma.masked_invalid(array)
ll, tt = np.meshgrid(x, y)
l1 = ll[~array.mask]
t1 = tt[~array.mask]
newg = array[~array.mask]
grfi = interpolate.griddata((l1, t1), newg.ravel(), (ll, tt), method='linear')
grfi[np.isnan(grfi)] = 0

#sigma = .95
#grfi = gaussian_filter(grfi, sigma)

cm = 0.25
cl = np.linspace(.0, cm, 21)
#grf[np.logical_not(np.isfinite(grf))] = 0
plt.figure(figsize=(10, 8))
plt.contourf(a['ll']/(2*np.pi), time*a['f']/(2*np.pi), grfi/a['f'], cl, vmin=-cm, vmax=cm, cmap='RdGy_r')
cb = plt.colorbar()
cb.set_ticks(np.linspace(cl[0], cl[-1], 6))
cb.set_label('$\\omega/f$', fontsize = 26)
plt.ylim((0, 30))
CS = plt.contour(a['ll']/(2*np.pi), time*a['f']/(2*np.pi), grfi/a['f'],
            np.linspace(.025, cm, 4),colors='0.5')
plt.xlim((0, 0.00150))
#plt.tick_params(axis='both', which='major', labelsize=fs)
#mlocs = [(.003, 8), (.005, 10), (.003, 22),(0.007, 6)]
mlocs = np.array([(.003/(2*np.pi), 8), (.005/(2*np.pi), 10), (.003/(2*np.pi), 22),(0.007/(2*np.pi), 6)])
mlocs = [(0.001/(2*np.pi), 10)] # For Upwelling case
plt.clabel(CS, inline=1, fontsize = 10, fmt='%1.3f', manual = mlocs)
plt.xlabel('$l$ $[m^{-1}]$', fontsize = 24)
plt.ylabel('$\\frac{tf}{2\\pi}$', fontsize = 26)
plt.grid(linestyle='--', alpha = 0.5)

#plt.savefig('/home/jacob/Dropbox/Slope BI/Slope BI Manuscript/EkmanStabilityDOWN.eps', format='eps', dpi=1000)
#plt.savefig('/home/jacob/Dropbox/Slope BI/Slope BI Manuscript/EkmanStabilityUP.eps', format='eps', dpi=1000)

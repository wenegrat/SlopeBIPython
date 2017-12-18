#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 09:34:19 2017

@author: jacob
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# LOAD
a = np.load('/home/jacob/dedalus/MixingStabilityOut.npz');
kap = a['kap']
z = a['z']
U = a['U']
V = a['V']
N = a['N']
B = a['B']
Bz = a['Bz']

tht = a['tht']
ll = a['ll']
gr = a['gr']
BP = a['BP']
SP = a['SP']
DISS = a['DISS']
nz = a['nz']
u = a['u']
v = a['v']
w = a['w']
b = a['b']

#Vz = np.diff(V)/np.diff(z)
#Uz = np.diff(U)/np.diff(z)
Vz = a['Vz']
Uz = a['Uz']

MagSh = np.sqrt((Vz*np.cos(tht))**2 + (Uz*np.cos(tht))**2)
Ri = (Bz*np.cos(tht)+N**2)/MagSh**2
# mean state

#%%
fs = 18

plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': fs})
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

fig, ax = plt.subplots(2, 2, sharey=True,sharex=False,figsize=(6, 8))



#ax[0].get_xaxis().set_label_coords(.5, -.12)

ax[0, 0].plot(U.real, z, linewidth=2)
ax[0, 0].plot(V.real, z, linewidth=2)
#ax[1].set_xlim((0, 0.1))
ax[0, 0].set_xticks([-0.1, -0.05, 0])
ax[0, 0].set_xlim((-0.1, .005))
ax[0, 0].set_ylim((0, 2500))
ax[0, 0].set_xlabel('Mean flow [m/s]', fontsize=fs)
#ax[1].get_xaxis().set_label_coords(.5, -.12)
ax[0, 0].grid(linestyle='--', alpha = 0.5)
ax[0, 0].set_ylabel('Slope-normal coordinate [m]', fontsize=fs)

ax[0, 1].plot(N**2*np.cos(tht)*z + B.real, z, linewidth=2)
ax[0, 1].set_xlabel('Mean buoyancy [m/s$^2$]', fontsize=fs)
#ax[2].get_xaxis().set_label_coords(.5, -.12)
ax[0, 1].set_xlim((0, 0.002))
ax[0, 1].set_ylim((0, 2500))
ax[0, 1].grid(linestyle='--', alpha = 0.5)

ax[1, 0].semilogx(kap, z, linewidth=2)
ax[1, 0].set_xticks([1e-5, 1e-4, 1e-3])
ax[1, 0].set_xlabel('Mixing coefficient [m$^2$/s]', fontsize=fs)
ax[1, 0].set_ylabel('Slope-normal coordinate [m]', fontsize=fs)
ax[1, 0].grid(linestyle='--', alpha = 0.5)
ax[1, 0].set_ylim((0, 2500))

ax[1,1].semilogx(Ri, z, linewidth=2)
ax[1,1].set_xticks(np.logspace(-1, 3, 5))
ax[1,1].set_xlabel('Richardson Number', fontsize=fs, labelpad=8)
#ax[3].set_ylabel('Slope-normal coordinate [m]', fontsize=fs)
ax[1,1].grid(linestyle='--', alpha = 0.5)
ax[1,1].set_ylim((0, 2500))
ax[1,1].set_xlim((1e-1, 1000))
plt.tight_layout()
#fig.savefig('/home/jacob/Dropbox/Slope BI/Slope BI Manuscript/MixingBasicState.pdf')



#%%
fs = 18

plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': fs})
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

fig, ax = plt.subplots(1, 3, sharey=True,figsize=(9, 5))

ax[0].semilogx(kap, z, linewidth=2)
ax[0].set_xticks([1e-5, 1e-4, 1e-3])
ax[0].set_xlabel('Mixing coefficient [m$^2$/s]', fontsize=fs)
ax[0].set_ylabel('Slope-normal coordinate [m]', fontsize=fs)
ax[0].grid(linestyle='--', alpha = 0.5)
ax[0].set_ylim((0, 2500))

#ax[0].get_xaxis().set_label_coords(.5, -.12)

ax[1].plot(U.real, z, linewidth=2)
ax[1].plot(V.real, z, linewidth=2)
#ax[1].set_xlim((0, 0.1))
ax[1].set_xticks([-0.1, -0.05, 0])
ax[1].set_xlim((-0.1, .005))
ax[1].set_ylim((0, 2500))
ax[1].set_xlabel('Mean flow [m/s]', fontsize=fs)
#ax[1].get_xaxis().set_label_coords(.5, -.12)
ax[1].grid(linestyle='--', alpha = 0.5)

ax[2].plot(N**2*np.cos(tht)*z + B.real, z, linewidth=2)
ax[2].set_xlabel('Mean buoyancy [m/s$^2$]', fontsize=fs)
#ax[2].get_xaxis().set_label_coords(.5, -.12)
ax[2].set_xlim((0, 0.002))
ax[2].set_ylim((0, 2500))
ax[2].grid(linestyle='--', alpha = 0.5)
plt.tight_layout()
#fig.savefig('/home/jacob/Dropbox/Slope BI/Slope BI Manuscript/MixingBasicState.pdf')

# energetics
#%%
fs = 20
plt.rcParams.update({'font.size': fs})
def tickfun(X):
    Y = 1/X/1000
    return ['%i' % ceil(z) for z in Y]

fig = plt.figure(figsize=(7, 4))
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()

ax1.semilogx(ll/(2*np.pi), gr, linewidth=2)
ax1.set_xlabel('Along-slope wavenumber [m$^{-1}$]', fontsize=fs)
ax1.set_ylabel('Growth rate [$s^{-1}$]', fontsize=fs)
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax1.grid(linestyle='--', alpha = 0.5, which='Both')
ax1.set_xlim((3e-6, 1e-3))
newticks = np.array([1/100e3, 1/10e3, 1/1e3])
ax2.set_xscale('log')
#
ax2.set_xticks(newticks)
ax2.set_xlim(ax1.get_xlim())
#
ax2.set_xticklabels(tickfun(newticks))
ax2.set_xlabel('Wavelength [km]', labelpad=8)
ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

plt.tight_layout()

#plt.savefig('/home/jacob/Dropbox/Slope BI/Slope BI Manuscript/MixingStability.pdf')


#%%
plt.figure(figsize=(5, 6))
plt.plot(BP/np.max(BP), z, linewidth=2)
plt.plot(SP/np.max(BP), z, linewidth=2)
plt.plot(DISS/np.max(BP), z, linewidth=2)
plt.xlabel('Kinetic energy tendency ')
plt.ylabel('Slope-normal coordinate [m]')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
leg = plt.legend([ 'Buoyancy production', 'Shear production', 'Dissipation'], frameon=True, loc=9)
leg.get_frame().set_alpha(.9)
plt.tight_layout()
plt.grid(linestyle='--', alpha = 0.5)
plt.ylim((0, 2500))
plt.xlim((-1.1, 1.1))
#plt.savefig('fig/energetics.pdf')
#plt.savefig('/home/jacob/Dropbox/Slope BI/Slope BI Manuscript/MixingEnergetics.pdf')
#%%
# most unstable mode

ly = np.linspace(0, 2*np.pi, nz)

fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(6.4, 6.4))
im = ax[0,0].pcolormesh(ly, z, np.real(u.reshape(nz, 1)
        * np.exp(1j*ly.reshape(1,nz))), rasterized=True, cmap='RdBu_r')
plt.colorbar(im, ax=ax[0,0])
ax[0,0].set_title('across-slope velocity')
im = ax[0,1].pcolormesh(ly, z, np.real(v.reshape(nz, 1)
        * np.exp(1j*ly.reshape(1,nz))), rasterized=True, cmap='RdBu_r')
plt.colorbar(im, ax=ax[0,1])
ax[0,1].set_title('along-slope velocity')
im = ax[1,0].pcolormesh(ly, z, np.real(w.reshape(nz, 1)
        * np.exp(1j*ly.reshape(1,nz))), rasterized=True, cmap='RdBu_r')
plt.colorbar(im, ax=ax[1,0])
ax[1,0].set_title('slope-normal velocity')
im = ax[1,1].pcolormesh(ly, z, np.real(b.reshape(nz, 1)
        * np.exp(1j*ly.reshape(1,nz))), rasterized=True, cmap='RdBu_r')
plt.colorbar(im, ax=ax[1,1])
ax[1,1].set_title('buoyancy')
ax[0,0].set_xticks([0, np.pi, 2*np.pi])
ax[1,0].set_xlabel('phase')
ax[1,1].set_xlabel('phase')
ax[0,0].set_ylabel('slope-normal coordinate [m]')
ax[1,0].set_ylabel('slope-normal coordinate [m]')
#plt.savefig('fig/modes.pdf', dpi=300)

plt.show()

#%%
#%%
# most unstable mode
fs =20
plt.rcParams.update({'font.size': fs})

nc  = 40
ly = np.linspace(0, 2*np.pi, nz)
uvel = np.real(u.reshape(nz, 1)* np.exp(1j*ly.reshape(1,nz)))
vvel = np.real(v.reshape(nz, 1)* np.exp(1j*ly.reshape(1,nz)))
wvel = np.real(w.reshape(nz, 1)* np.exp(1j*ly.reshape(1,nz)))
maxu = np.max(uvel)

uvel = uvel/maxu
vvel = vvel/maxu
wvel = wvel/maxu
buoy = np.real(b.reshape(nz, 1)* np.exp(1j*ly.reshape(1,nz)))
buoy = buoy/np.max(buoy)

fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 10))
# UVEL
im = ax[0,0].contourf(ly, z, uvel, np.linspace(-1, 1, nc),vmin=-1, vmax=1, cmap='RdBu_r')
cb = plt.colorbar(im, ax=ax[0,0])
cb.set_ticks([-1, 0, 1])
ax[0,0].set_title('Across-slope velocity', fontsize=fs)
ax[0,0].grid(linestyle='--', alpha = 0.5)

# VVEL
cl = 0.4
im = ax[0,1].contourf(ly, z, vvel, np.linspace(-cl, cl, nc),vmin=-cl, vmax=cl, cmap='RdBu_r')
cb = plt.colorbar(im, ax=ax[0,1])
cb.set_ticks([-cl, 0, cl])
ax[0,1].set_title('Along-slope velocity', fontsize=fs)
ax[0,1].grid(linestyle='--', alpha = 0.5)

# WVEL
cl = 0.02
im = ax[1,0].contourf(ly, z, wvel, np.linspace(-cl,cl, nc),vmin=-cl, vmax=cl, cmap='RdBu_r')
cb = plt.colorbar(im, ax=ax[1,0])
cb.set_ticks([-cl, 0, cl])
ax[1,0].set_title('Slope-normal velocity', fontsize=fs)
ax[1,0].grid(linestyle='--', alpha = 0.5)

# BUOY
im = ax[1,1].contourf(ly, z, buoy, np.linspace(-1, 1, nc),vmin=-1, vmax=1, cmap='RdBu_r')
cb = plt.colorbar(im, ax=ax[1,1])
ax[1,1].grid(linestyle='--', alpha = 0.5)
cb.set_ticks([-1, 0, 1])
ax[1,1].set_title('Buoyancy', fontsize=fs)
ax[0,0].set_xticks([0, np.pi, 2*np.pi])
ax[1,0].set_xlabel('Phase', fontsize=fs)
ax[1,1].set_xlabel('Phase', fontsize=fs)
ax[0,0].set_ylabel('Slope-normal coordinate [m]', fontsize=fs)
ax[1,0].set_ylabel('Slope-normal coordinate [m]', fontsize=fs)

#labels = [item.get_text() for item in ax[1,1].get_xticklabels()]
labels = ['0', '$\pi$', '$2\pi$']
ax[1,1].set_xticklabels(labels)  
#plt.savefig('fig/modes.pdf', dpi=300)
plt.tight_layout()
#plt.savefig('/home/jacob/Dropbox/Slope BI/Slope BI Manuscript/MixingPerturbations.pdf', format='pdf')

plt.show()
#%%
#%%
cm = 'RdBu_r'
# most unstable mode
fs = 18

plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': fs})
plt.rcParams['contour.negative_linestyle'] = 'solid'

nc  = 16
lw =0.5
ly = np.linspace(0, 2*np.pi, nz)
uvel = np.real(u.reshape(nz, 1)* np.exp(1j*ly.reshape(1,nz)))
vvel = np.real(v.reshape(nz, 1)* np.exp(1j*ly.reshape(1,nz)))
wvel = np.real(w.reshape(nz, 1)* np.exp(1j*ly.reshape(1,nz)))
maxu = np.max(uvel)

uvel = uvel/maxu
vvel = vvel/maxu
wvel = wvel/maxu
buoy = np.real(b.reshape(nz, 1)* np.exp(1j*ly.reshape(1,nz)))
buoy = buoy/np.max(buoy)

fig = plt.figure(figsize=(10, 9))
gs = GridSpec(4, 5)
gs.update(hspace=0.3, wspace=.45)
ax3 = fig.add_subplot(gs[2:, :-3])

ax1 = fig.add_subplot(gs[:-2, :-3])
ax2 = fig.add_subplot(gs[:-2, 2:-1])
ax4 = fig.add_subplot(gs[2:, 2:-1])

gs2 = GridSpec(4,1)
gs2.update(left=0.875, right=1.1)
ax5 = fig.add_subplot(gs2[1:-1, 0])

#ax1 = plt.subplot2grid((4, 5), (0, 0), colspan=2, rowspan=2)
#ax2 = plt.subplot2grid((4, 5 ),(0, 2), colspan=2, rowspan=2)
#ax3 = plt.subplot2grid((4, 5), (2, 0), colspan=2, rowspan=2)
#ax4 = plt.subplot2grid((4, 5), (2, 2), colspan=2, rowspan=2)
#ax5 = plt.subplot2grid((4, 5), (1, 4), rowspan=2)


#fig = 
# UVEL
im = ax1.contourf(ly, z, uvel, np.linspace(-1, 1, nc),vmin=-1, vmax=1, cmap=cm)
cb = plt.colorbar(im, ax=ax1)
cb.set_ticks([-1, 0, 1])
ax1.contour(ly, z, uvel, np.linspace(-1, 1, nc), colors='0.8', linewidths=lw)
ax1.set_title('Across-slope velocity', fontsize=fs)
ax1.grid(linestyle='--', alpha = 0.5)

# VVEL
im = ax2.contourf(ly, z, vvel, np.linspace(-0.4, 0.4, nc),vmin=-0.4, vmax=0.4, cmap=cm)
cb = plt.colorbar(im, ax=ax2)
cb.set_ticks([-0.4, 0, 0.4])
ax2.contour(ly, z, vvel, np.linspace(-0.4, 0.4, nc), colors='0.8', linewidths=lw)

ax2.set_title('Along-slope velocity', fontsize=fs)
ax2.grid(linestyle='--', alpha = 0.5)

# WVEL
im = ax3.contourf(ly, z, wvel, np.linspace(-0.02, 0.02, nc),vmin=-0.02, vmax=0.02, cmap=cm)
cb = plt.colorbar(im, ax=ax3)
cb.set_ticks([-0.02, 0, 0.02])
ax3.contour(ly, z, wvel, np.linspace(-0.02, 0.02, nc), colors='0.8', linewidths=lw)

ax3.set_title('Slope-normal velocity', fontsize=fs)
ax3.grid(linestyle='--', alpha = 0.5)

# BUOY
im = ax4.contourf(ly, z, buoy, np.linspace(-1, 1, nc),vmin=-1, vmax=1, cmap=cm)
cb = plt.colorbar(im, ax=ax4)
ax4.grid(linestyle='--', alpha = 0.5)
ax4.contour(ly, z, buoy, np.linspace(-1, 1, nc), colors='0.8', linewidths=lw)

np.linspace(-0.15, 0.15, nc)
cb.set_ticks([-1, 0, 1])
ax4.set_title('Buoyancy', fontsize=fs)
ax1.set_xticks([0, np.pi, 2*np.pi])
ax2.set_xticks([0, np.pi, 2*np.pi])
ax3.set_xticks([0, np.pi, 2*np.pi])
ax4.set_xticks([0, np.pi, 2*np.pi])

ax3.set_xlabel('Along-slope phase', fontsize=fs)
ax4.set_xlabel('Along-slope phase', fontsize=fs)
ax1.set_ylabel('Slope-normal coordinate [m]', fontsize=fs)
ax3.set_ylabel('Slope-normal coordinate [m]', fontsize=fs)


# Energetics
ax5.plot(BP/np.max(BP), z)
ax5.plot((SP)/np.max(BP), z)
ax5.plot(DISS/np.max(BP), z)
#plt.plot(LSP/np.max(BP), z)

ax5.set_xlabel('Kinetic energy tendency', fontsize=fs)
ax5.set_ylabel('Slope-normal coordinate [m]', fontsize=fs)
ax5.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
leg = ax5.legend(['Buoyancy Production', 'Shear Production', 'Dissipation'], frameon=True, fontsize=12, loc=9)
leg.get_frame().set_alpha(.9)
#plt.tight_layout()
ax5.set_ylim((0, 2500))
ax5.set_xlim((-1.1, 1.1))
ax5.grid(linestyle='--', alpha = 0.5)


#ax3.set_xlim((0, 2*np.pi))
#labels = [item.get_text() for item in ax[1,1].get_xticklabels()]
labels = ['0', '$\pi$', '$2\pi$']
ax3.set_xticklabels(labels)  
ax4.set_xticklabels(labels)
emptylabels = ['']*len(ax3.get_xticklabels())
ax1.set_xticklabels(emptylabels)
ax2.set_xticklabels(emptylabels)
emptylabels = ['']*len(ax1.get_yticklabels())
ax2.set_yticklabels(emptylabels)
ax4.set_yticklabels(emptylabels)
#plt.savefig('fig/modes.pdf', dpi=300)
#plt.tight_layout()
#plt.savefig('/home/jacob/Dropbox/Slope BI/Slope BI Manuscript/MixingPerturbationsCombo.pdf', format='pdf', bbox_inches='tight')
#fig.subplots_adjust(wspace=10, vspace=0.1)
plt.show()
#%% ADDING MY OWN CHECK OF RI
#Bz = np.diff(B)/np.diff(z)
f = 2.5e-5
N2hat = Bz + N**2*np.cos(tht)
M2hat = N**2*np.sin(tht)
#N2 = N2hat*np.cos(tht) + M2hat*np.sin(tht)
#M2 = M2hat*np.cos(tht) - N2hat*np.sin(tht)
N2 = N**2 + Bz*np.cos(tht)
M2 = -Bz*np.sin(tht)

Ri = N2*f**2/(M2**2)
#dt = np.sqrt(2*kap_1/f)
#S = N2/(f**2)* np.tan(tht)**2
#delt = np.sqrt(S*Ri)
#delt = N2hat/M2hat*np.tan(tht)
#thtiso = np.arctan(M2/N2)
#thtiso = np.arctan(M2hat/N2hat)

#iso = np.zeros((M2.shape[0], 2))
#iso[:,0] = M2
#iso[:,1] = N2
#sl = np.zeros((M2.shape[0], 2))
#sl[:,0] = 1
#sl[:,1] = np.tan(tht)

plt.figure()
#plt.plot(-N**2/Bz - 1, z)
plt.plot(Ri , z)
#plt.plot(Bz*z, z)
#plt.axhline(y=dt, color='r', linestyle='-')
#plt.ylim((0, 500))
plt.xlim((0, 1000))
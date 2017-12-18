#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 15:41:53 2017

Plot the Mechoso 1980 solutions.
@author: jacob
"""
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
import pylab as pylab
nll = 1000
ndelt = 500

N = 1e-3
H = 1000
f = 1e-4
Ri = 100

Rd = N*H/f

ll = np.linspace(0, 5, num=nll)

# First do flat top version
delta = np.linspace(-3, 3, num=ndelt)
deltat = delta

omega = np.zeros((ndelt, nll), np.complex128)
omegas = np.zeros((ndelt, nll), np.complex128)

for i in range(0, ll.shape[0]):
    l = ll[i]
    omega[:,i] = 1/Ri**(1/2)*np.sqrt( (l-(1+0*deltat)*np.tanh(l))/np.tanh(l)*(1+delta)
                 - 1/4*( (0*deltat -delta)/np.tanh(l) - l)**(2) + 0j)
    omegas[:,i] = 1/Ri**(1/2)*np.sqrt( (l-(1+deltat)*np.tanh(l))/np.tanh(l)*(1+delta)
         - 1/4*( (deltat -delta)/np.tanh(l) - l)**(2) + 0j)
om = omega.real
oms = omegas.real

# Make Figure
#%%
plt.rcParams['text.usetex'] = True
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
nc = 41
maxc = 0.05
fs =18
fig, ax = plt.subplots(1, 2, sharey=True, figsize=(12, 7))

ax[0].grid(linestyle='--', alpha = 0.5)
ax[0].contourf(ll, delta, om,
      np.linspace(0, maxc, nc),vmin=-maxc, vmax=maxc, cmap='RdBu_r', labelsize=20)
CS = ax[0].contour(ll, delta, om, np.linspace(0, maxc, 6),colors='0.5' )
ax[0].set_yticks([-2, -1, 0, 1, 2])
ax[0].tick_params(axis='both', which='major', labelsize=fs)
ax[0].clabel(CS, inline=1, fontsize=10, fmt='%1.2f')
ax[0].set_ylabel('Slope parameter, $\\alpha$', fontsize=20)
ax[0].set_xlabel('Along-slope wavenumber, $l^*$', fontsize=20)
ax[0].set_ylim((-2,2))
ax[0].set_xlim((0, 4))
bb = dict(boxstyle='Square', fc='w')
ax[0].text(0.15, 1.65, '$\\alpha^{ub} = 0$', fontsize=20, bbox=bb)

ax[1].grid(linestyle='--', alpha = 0.5)
IM = ax[1].contourf(ll, delta, oms,
      np.linspace(0, maxc, nc),vmin=-maxc, vmax=maxc, cmap='RdBu_r', labelsize=20)
CS = ax[1].contour(ll, delta, oms, np.linspace(0, maxc, 6),colors='0.5' )
ax[1].clabel(CS, inline=1, fontsize=10, fmt='%1.2f')
ax[1].set_xlabel('Along-slope wavenumber, $l^*$', fontsize=20)
ax[1].tick_params(axis='both', which='major', labelsize=fs)
ax[1].set_ylim((-2,2))
ax[1].set_xlim((0, 4))
ax[1].text(0.15, 1.65, '$\\alpha^{ub} = \\alpha$', fontsize=20, bbox=bb)
plt.tight_layout()

cbar = fig.colorbar(IM,  ax=ax.ravel().tolist(), orientation='horizontal', shrink = 0.5)
cbar.set_label('Growth rate, ${\omega}_i$', fontsize=20)
cbar.set_ticks(np.linspace(0, maxc, 6))
plt.savefig('/home/jacob/Dropbox/Slope BI/Slope BI Manuscript/QGPlot.eps', format='eps', dpi=1000)
#%%
#plt.figure()
#plt.plot(ll, om[250,:])
#plt.plot(ll, om[251,:])

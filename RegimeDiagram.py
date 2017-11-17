#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 09:21:41 2017

Try to create a regime diagram
@author: jacob
"""
import numpy as np
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)

ns = 100
nri = 200
maxr = 10
maxs = 3
S = np.linspace(0, maxs, ns)
Ri =np.transpose(np.linspace(0, maxr, nri))
Rif = np.zeros((ns, nri))
Sf = np.zeros((ns, nri))
for i in range(0, ns):
    Rif[i,:] = np.linspace(0, maxr, nri)
for i in range(0, nri):
    Sf[:,i] = np.linspace(0, maxs, ns)
#Rif = np.outer(Ri, Ri)plt
delta = (Sf*Rif)**(1/2)
Pr = .1
SICI = (1/delta)*(1-Pr*(1-delta/Rif))**(-1)
BICI = delta
CIBI = delta*(1+Rif)/(Rif)
CCRIT = delta/Rif
SICRIT = Rif 

#plt.figure()
##plt.contourf(Ri, S, delta)
##plt.contourf(Ri, S, delta, np.linspace(0, 10, 11))
##plt.contour(Ri, S,CIBI, levels=[1], colors='k')
#plt.contour(Ri, S,CCRIT, levels=[1], colors='r', linestyles='dashed')
##plt.contour(Ri, S,SICRIT, levels=[1], colors='g', linestyles='dashed')
#plt.contour(Ri, S,delta, levels=[1, 2], colors='k', linestyles='dotted')
##plt.contour(Ri, S,delta, levels=[2], colors='b', linestyles='dashed')
#plt.contour(Ri, S,(1/Rif)*(1+delta), levels=[1], colors='b', linestyles='dashed')
#plt.contour(Ri, S,SICI, levels=[1], colors='r', linestyles='dashed')


SICRIT = (1/Rif)*(1+delta)>1
VSHS = (1/delta)*(1-Pr*(1-delta/Rif))**(-1) > 1
CCRIT = delta/Rif > 1
maskSI = np.nan*np.zeros((Rif.shape))
maskSI[VSHS&SICRIT] = 1
maskCI = np.nan*np.zeros((Rif.shape))
#VSHSn = not VSHS
maskCI[CCRIT& np.logical_not(VSHS)] = 1

maskMI = np.nan*np.zeros((Rif.shape))
maskMI[SICRIT& np.logical_not(VSHS)&np.logical_not(CCRIT)] = 1
maskPV = np.nan*np.zeros((Rif.shape))
maskPV[np.logical_not(SICRIT)] = 1

maskVS = np.nan*np.zeros((Rif.shape))
maskVS[SICRIT] = 1
#maskSI[maskSI!=1] = NaN
cl = 8
plt.figure()
plt.contourf(Ri, S, maskSI*4, np.linspace(0, cl, 10), cmap='Greys')
plt.contourf(Ri, S, maskCI*3, np.linspace(0,cl, 10), cmap='Greys')
plt.contourf(Ri, S, maskMI*2, np.linspace(0, cl, 10), cmap='Greys')
plt.contourf(Ri, S, maskPV*1, np.linspace(0, cl, 10), cmap='Greys')
plt.contour(Ri, S,delta/Rif, levels=[1], colors='green', linestyles='dashed') 
CL = plt.contour(Ri, S,delta*maskPV, levels=[1, 2], colors='k', linestyles='dotted')
plt.clabel(CL, inline=1, fontsize = 12, fmt='$\delta$ $=$ %1.0f')
plt.contour(Ri, S,SICI*maskVS, levels=[1], colors='b', linestyles='dashed')
plt.contour(Ri, S,(1/Rif)*(1+delta), levels=[1], colors='r', linestyles='dashed')
plt.plot(1.25, 2.75, marker='x', color='k')
plt.plot(0.5, 0.1, marker='x', color='k')
plt.xlabel('Ri')
plt.ylabel('S')

plt.annotate(r'BI',
             xy=(4, 0.5), xycoords='data', xytext=(+0, +0), 
             textcoords='offset points', fontsize=16)
plt.annotate(r'SI',
             xy=(0.75, 0.35), xycoords='data', xytext=(+0, +0), 
             textcoords='offset points', fontsize=16)
plt.annotate(r'CI',
             xy=(1, 2.25), xycoords='data', xytext=(+0, +0), 
             textcoords='offset points', fontsize=16)
plt.annotate(r'SI/CI',
             xy=(2, 1.65), xycoords='data', xytext=(+0, +0), 
             textcoords='offset points', fontsize=16)
#plt.colorbar()
#plt.contour(Ri, S,Rif*(1-delta/Rif), levels=[1], colors='k', linestyles='dashed')

#plt.colorbar()

#plt.figure()
#plt.contourf(Ri, S, np.log10(SICI), 30)
#plt.xlabel('Ri')
#plt.ylabel('S')
#plt.colorbar()

#plt.savefig('/home/jacob/Dropbox/Slope BI/Slope BI Manuscript/RegimeDiagram.eps', format='eps', dpi=1000)

#%%
plt.figure()
phi = 1/2*np.arctan( (2*Pr*Rif**(-1/2))/(1 - Pr**2*(1-delta/Rif))) 
phit = 1/2*( (2*Pr*Rif**(-1/2))/(1 - Pr**2*(1-delta/Rif)))
plt.contourf(Ri, S, phi)
plt.colorbar()

plt.figure()
plt.plot(np.tan(phi[1,:]))
plt.plot(phit[1,:])
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

ns = 200
nri = 400
maxr = 5
maxs = 2
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
SICI = (1/delta)*(1-Pr*(1-delta/Rif))**(-1) # INVERSE DEFINITION FROM MS, IE VSP/LSP
BICI = delta
#CIBI = delta*(1+Rif)/(Rif) 
CCRIT = delta/Rif
SICRIT = Rif 


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

plt.rcParams['text.usetex'] = True
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 16})
cl = 8
plt.figure(figsize=(6, 4))
plt.contourf(Ri, S, maskSI*4, np.linspace(0, cl, 10), cmap='Greys')
plt.contourf(Ri, S, maskCI*3, np.linspace(0,cl, 10), cmap='Greys')
plt.contourf(Ri, S, maskMI*2, np.linspace(0, cl, 10), cmap='Greys')
plt.contourf(Ri, S, maskPV*1, np.linspace(0, cl, 10), cmap='Greys')
plt.contour(Ri, S,delta/Rif, levels=[1], colors='green', linestyles='dashed') 
CL = plt.contour(Ri, S,delta*maskPV, levels=[1, 2], colors='k', linestyles='dotted')
mlocs = [(4.5, 0.25), (4.65, 0.9)]
plt.clabel(CL, inline=1, fontsize = 12, fmt='$\\alpha$ $=$ %1.0f', manual=mlocs)
plt.contour(Ri, S,SICI*maskVS, levels=[1], colors='b', linestyles='dashed')
plt.contour(Ri, S,(1/Rif)*(1+delta), levels=[1], colors='r', linestyles='dashed')
#plt.plot(1.25, 2.75, marker='x', color='k')
#plt.plot(0.5, 0.1, marker='x', color='k')
plt.xlabel('Richardson number, Ri')
plt.ylabel('Slope Burger number, S')
plt.title('Instability regimes in the BBL')

plt.annotate(r'BI',
             xy=(3.25, 0.65), xycoords='data', xytext=(+0, +0), 
             textcoords='offset points', fontsize=16)
plt.annotate(r'SI',
             xy=(0.75, 0.35), xycoords='data', xytext=(+0, +0), 
             textcoords='offset points', fontsize=16)
plt.annotate(r'CI',
             xy=(.95, 1.65), xycoords='data', xytext=(+0, +0), 
             textcoords='offset points', fontsize=16)
plt.annotate(r'SI/CI',
             xy=(1.75, 1.25), xycoords='data', xytext=(+0, +0), 
             textcoords='offset points', fontsize=16)
# PLOT NUMERIC SOLS
maskPVNum = np.nan*np.zeros(LSP.shape)
maskPVNum[PV<0] = 1
maskPVNum[CI<1] = np.nan
plt.contour(rivec, svec, np.transpose(LSP/VSP*maskPVNum), levels=[1], colors='b')

plt.tight_layout()
#plt.ylim((0, 3))
#plt.xlim((0, 10))
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
#plt.figure()
#phi = 1/2*np.arctan( (2*Pr*Rif**(-1/2))/(1 - Pr**2*(1-delta/Rif))) 
#phit = 1/2*( (2*Pr*Rif**(-1/2))/(1 - Pr**2*(1-delta/Rif)))
#plt.contourf(Ri, S, (1/delta)*(1-Pr*(1-delta/Rif))**(-1), np.linspace(0, 5, 20))
#plt.colorbar()
#plt.ylim((0,2))
#plt.xlim((0, 5))
##plt.figure()
##plt.plot(np.tan(phi[1,:]))
##plt.plot(phit[1,:])
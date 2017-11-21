#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 14:38:04 2017

@author: jacob
"""

import sys
sys.path.append('/usr/share/')
import os
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import scipy.integrate as integrate

plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 18})
directoryname = "../SIRegimeScan/"
directory = os.fsencode(directoryname)

nri = 64
ns = 64

counter = 0
rivec = np.linspace(1e-2, 5, nri)
svec = np.linspace(0, 2, ns)

LSP = np.zeros((nri, ns), dtype=np.float64)    
VSP = np.zeros((nri, ns), dtype=np.float64)    
BP = np.zeros((nri, ns), dtype=np.float64)    
HHF = np.zeros((nri, ns), dtype=np.float64)    
Ri = np.zeros((nri, ns), dtype=np.float64)    
S = np.zeros((nri, ns), dtype=np.float64)    
PV = np.nan*np.zeros((nri, ns), dtype=np.float64)    
delta = np.zeros((nri, ns), dtype=np.float64)    
rif = np.zeros((nri, ns), dtype = np.float64)
sif = np.zeros((nri, ns), dtype=np.float64)
CI = np.nan*np.zeros((nri, ns), dtype=np.float64)    
GR = np.zeros((nri, ns), dtype=np.float64)    
GRR = np.zeros((nri, ns), dtype=np.float64)    

plt.figure()
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    print(filename)
    if filename.endswith(".npz"): 
        a = np.load(directoryname+filename);
        rit = a['Ri']
        sit = a['S']
        rind = np.where(rivec==rit)[0][0]
        sind = np.where(svec ==sit)[0][0]
        LSP[rind, sind] = integrate.trapz(a['LSP'],a['z'])#[:,-1]
        VSP[rind, sind] = integrate.trapz(a['VSP'],a['z'])#[:,-1]
#        LSP[rind, sind] = np.max(a['LSP'])
#        VSP[rind, sind] = np.max(a['VSP'])#[:,-1]
        BP[rind, sind] = integrate.trapz(a['BP'],a['z'])#[:,-1]
        HHF[rind, sind] = integrate.trapz(a['HHF'],a['z'])#[:,-1]
        GR[rind, sind] = a['gr']
        GRR[rind, sind] = a['grr']/a['k']
        PV[rind, sind] = 1 - 1/rit*(1+np.sqrt(sit*rit))
        delta[rind, sind] =np.sqrt(sit*rit)
        rif[rind, sind] = rit
        sif[rind, sind] = sit
        CI[rind, sind] = np.sqrt(sit*rit)/rit
#        plt.plot(a['ll'], gr[counter-1,:]*np.sqrt(a['Bz'][-1])/(a['f']*a['Vz'][-1]))
        continue
    else:
        continue
    
    
#%%
cm = 'seismic'
ET = HHF + LSP + VSP
cl = np.append(np.append(-100, np.linspace(-1, 1, 50)*4e-3), 100)
fig, ax = plt.subplots(1, 3, figsize=(10, 4), sharey=True)
ax[0].contourf(rivec, svec, np.transpose((BP)), cl, cmap=cm, vmin=cl[1], vmax=cl[-2])
ax[0].contour(rivec, svec, np.transpose(PV), levels=[0], colors='r')
ax[0].contour(rivec, svec, np.transpose(CI), levels=[1], colors='g')
#plt.contour(rivec, svec, np.transpose(LSP/VSP), levels=[1])
ax[0].set_title('HHF')
ax[0].contour(rivec, svec, delta, levels=[1], colors='k')


ax[1].contourf(rivec, svec, np.transpose((LSP)), cl, cmap=cm, vmin=cl[1], vmax=cl[-2])
ax[1].contour(rivec, svec, np.transpose(PV), levels=[0], colors='r')
ax[1].contour(rivec, svec, np.transpose(CI), levels=[1], colors='g')
#plt.contour(rivec, svec, np.transpose(LSP/VSP), levels=[1])
ax[1].set_title('LSP')
ax[1].contour(rivec, svec, delta, levels=[1], colors='k')
#ax[0].set_ylim((0, 1))
IM = ax[2].contourf(rivec, svec, np.transpose((VSP)), cl, cmap=cm, vmin=cl[1], vmax=cl[-2])
ax[2].contour(rivec, svec, np.transpose(PV), levels=[0], colors='r')
ax[2].contour(rivec, svec, np.transpose(CI), levels=[1], colors='g')
#plt.contour(rivec, svec, np.transpose(LSP/VSP), levels=[1])
ax[2].set_title('VSP')
ax[2].contour(rivec, svec, np.transpose(delta), levels=[1], colors='k')
#ax[2].contour(rivec, svec, np.transpose(KET), levels=[0], colors = 'b')
cbar = fig.colorbar(IM,  ax=ax.ravel().tolist(), orientation='horizontal', shrink = 1)
cbar.set_ticks(np.linspace(cl[1], cl[-2], 3))

ax[0].set_ylim((0, 1))
ax[0].set_xlim((0, 5))
ax[1].set_xlim((0, 5))
ax[2].set_xlim((0, 5))
#%% COMPARE TO ET
cm = 'seismic'
ET = HHF + LSP + VSP
KET = BP + LSP + VSP
mask = np.zeros(ET.shape)
mask[PV<0] = 1
maskdom = np.zeros(ET.shape)
maskdom[(LSP/ET > VSP/ET)] = 1.001
maskdom[VSP<0] = 0
maskKE = np.zeros(ET.shape)
maskKE[KET>0] = 1.0001
maskGR = np.zeros(ET.shape)
maskGR[GR/a['f']>0.01] = 1.0001
cl = np.append(np.append(-100, np.linspace(-1, 1, 50)*2), 100)
fig, ax = plt.subplots(1, 3, figsize=(10, 4), sharey=True)
ax[0].contourf(rivec, svec, np.transpose((HHF)/ET*mask), cl, cmap=cm, vmin=cl[1], vmax=cl[-2])
ax[0].contour(rivec, svec, np.transpose(PV), levels=[0], colors='r')
ax[0].contour(rivec, svec, np.transpose(CI), levels=[1], colors='g')
#plt.contour(rivec, svec, np.transpose(LSP/VSP), levels=[1])
ax[0].set_title('$HHF/(KE + PE)_t$')
#ax[0].contour(rivec, svec, delta, levels=[1], colors='k')


ax[1].contourf(rivec, svec, np.transpose((LSP)/ET*mask), cl, cmap=cm, vmin=cl[1], vmax=cl[-2])
ax[1].contour(rivec, svec, np.transpose(PV), levels=[0], colors='r')
ax[1].contour(rivec, svec, np.transpose(CI), levels=[1], colors='g')
#plt.contour(rivec, svec, np.transpose(LSP/VSP), levels=[1])
ax[1].set_title('$LSP/(KE + PE)_t$')
#ax[1].contour(rivec, svec, delta, levels=[1], colors='k')
#ax[0].set_ylim((0, 1))
IM = ax[2].contourf(rivec, svec, np.transpose((VSP)/ET*mask), cl, cmap=cm, vmin=cl[1], vmax=cl[-2])
ax[2].contour(rivec, svec, np.transpose(PV), levels=[0], colors='r')
ax[2].contour(rivec, svec, np.transpose(CI), levels=[1], colors='g')
#plt.contour(rivec, svec, np.transpose(LSP/VSP), levels=[1])
ax[2].set_title('$VSP/(KE + PE)_t$')
ax[2].contour(rivec, svec, np.transpose(delta), levels=[1], colors='k')
ax[2].contour(rivec, svec, np.transpose(maskdom*mask), levels=[1], colors='b')
ax[2].contour(rivec, svec, np.transpose(maskKE*mask), levels=[1])
ax[0].contour(rivec, svec, np.transpose(maskKE*mask), levels=[1])
ax[1].contour(rivec, svec, np.transpose(maskKE*mask), levels=[1])
ax[0].set_ylabel('S')
ax[0].set_xlabel('Ri')
ax[1].set_xlabel('Ri')
ax[2].set_xlabel('Ri')

#ax[2].contour(rivec, svec, np.transpose(maskGR), levels=[1], colors='b')
#ax[2].contour(rivec, svec, np.transpose(KET), levels=[0], colors = 'b')
cbar = fig.colorbar(IM,  ax=ax.ravel().tolist(), orientation='horizontal', shrink = 0.5, pad=0.25)
cbar.set_ticks(np.linspace(cl[1], cl[-2], 3))
#plt.tight_layout()
ax[0].set_ylim((0, 2))
ax[0].set_xlim((0, 5))
ax[1].set_xlim((0, 5))
ax[2].set_xlim((0, 5))

#%% NORMALIZED BY KET
KET = BP + LSP + VSP
#KET = np.abs(KET)
cl = np.linspace(-1, 1, 10)*2
fig, ax = plt.subplots(1, 3, figsize=(10, 6))
ax[0].contourf(rivec, svec, np.transpose((BP/KET)), cl)
ax[0].contour(rivec, svec, np.transpose(PV), levels=[0], colors='r')
ax[0].contour(rivec, svec, np.transpose(CI), levels=[1], colors='g')
#plt.contour(rivec, svec, np.transpose(LSP/VSP), levels=[1])
ax[0].set_title('BP')
ax[0].contour(rivec, svec, delta, levels=[1], colors='k')

ax[1].contourf(rivec, svec, np.transpose((LSP/KET)), cl)
ax[1].contour(rivec, svec, np.transpose(PV), levels=[0], colors='r')
ax[1].contour(rivec, svec, np.transpose(CI), levels=[1], colors='g')
#plt.contour(rivec, svec, np.transpose(LSP/VSP), levels=[1])
ax[1].set_title('LSP')
ax[1].contour(rivec, svec, delta, levels=[1], colors='k')

IM = ax[2].contourf(rivec, svec, np.transpose((VSP/KET)), cl)
ax[2].contour(rivec, svec, np.transpose(PV), levels=[0], colors='r')
ax[2].contour(rivec, svec, np.transpose(CI), levels=[1], colors='g')
#plt.contour(rivec, svec, np.transpose(LSP/VSP), levels=[1])
ax[2].set_title('VSP')
ax[2].contour(rivec, svec, delta, levels=[1], colors='k')

cbar = fig.colorbar(IM,  ax=ax.ravel().tolist(), orientation='horizontal', shrink = 1)
cbar.set_ticks(np.linspace(cl[0], cl[-1], 3))
#%%
plt.contourf(rivec, svec, np.transpose(GR/a['f']))
plt.contour(rivec, svec, np.transpose(GR/a['f']), levels=[.000001], colors='g')
plt.contour(rivec, svec, np.transpose(PV), levels=[0], colors='r')

#plt.contourf(rivec, svec, np.transpose(GRR))
plt.colorbar()


#plt.contour(rivec, svec, np.transpose(PV), levels=[0], colors='r')

#%%
cl = np.linspace(-1, 1, 10)*2
fig, ax = plt.subplots(1, 2, figsize=(10, 6))
ax[0].contourf(rivec, svec, np.transpose((BP/VSP)), cl)
ax[0].contour(rivec, svec, np.transpose(PV), levels=[0], colors='r')
ax[0].contour(rivec, svec, np.transpose(CI), levels=[1], colors='g')
#plt.contour(rivec, svec, np.transpose(LSP/VSP), levels=[1])
ax[0].set_title('BP/VSP')
ax[0].contour(rivec, svec, delta, levels=[1], colors='k')
#ax[0].contour(rivec, svec, delta, levels=[1/2], colors='b')

IM = ax[1].contourf(rivec, svec, np.transpose((LSP/VSP)), cl)
ax[1].contour(rivec, svec, np.transpose(PV), levels=[0], colors='r')
ax[1].contour(rivec, svec, np.transpose(CI), levels=[1], colors='g')
plt.contour(rivec, svec, np.transpose(LSP/VSP), levels=[1])
ax[1].set_title('LSP/VSP')
ax[1].contour(rivec, svec, delta, levels=[1], colors='k')
#ax[1].contour(rivec, svec, delta, levels=[1/2], colors='b')

cbar = fig.colorbar(IM,  ax=ax.ravel().tolist(), orientation='horizontal', shrink = 1)
cbar.set_ticks(np.linspace(cl[0], cl[-1], 3))

#%%
Pr = 0.1
#SICRIT = (1/Rif)*(1+delta)>1
VSHS = (delta)*(1-Pr**2*(1-delta/rif))
#CCRIT = delta/Rif > 1
#maskSI = np.nan*np.zeros((Rif.shape))
#maskSI[VSHS&SICRIT] = 1
#maskCI = np.nan*np.zeros((Rif.shape))
##VSHSn = not VSHS
#maskCI[CCRIT& np.logical_not(VSHS)] = 1
#
#maskMI = np.nan*np.zeros((Rif.shape))
#maskMI[SICRIT& np.logical_not(VSHS)&np.logical_not(CCRIT)] = 1
#
#maskVS = np.nan*np.zeros((Rif.shape))
#maskVS[SICRIT] = 1

ET = HHF + LSP + VSP
KET = BP + LSP + VSP
mask = np.zeros(ET.shape)
mask[PV<0] = 1
maskLSP = np.nan*np.zeros(ET.shape)
maskLSP[(LSP/ET >= VSP/ET)] = 1.001
maskLSP[VSP<0] = np.nan
maskLSP[PV>0] = np.nan

maskVSP = np.nan*np.zeros(ET.shape)
maskVSP[(LSP/ET <= VSP/ET) ] = 1.001
maskVSP[VSP<0] = np.nan
maskVSP[PV>0] = np.nan

maskPV = np.nan*np.zeros((PV.shape))
maskPV[np.logical_not(PV > 0)] = 1
maskPPV = np.nan*np.zeros((PV.shape))
maskPPV[PV>0] = 1
#maskSI[maskSI!=1] = NaN
cl = 8
plt.figure()
#IM = ax[2].contourf(rivec, svec, np.transpose((VSP)/ET*mask), cl, cmap=cm, vmin=cl[1], vmax=cl[-2])
plt.contourf(rivec, svec, np.transpose(maskPV*1), np.linspace(0, cl, 10), cmap='Greys') # Fill zero PV side
plt.contour(rivec, svec, np.transpose(PV), levels=[0], colors='r', linestyles='dashed') # Contour PV = 0 Line

plt.contour(rivec, svec, np.transpose(CI), levels=[1], colors='g', linestyles='dashed') # Contour the zero absolute vertical vorticity line
#plt.contour(rivec, svec, np.transpose(LSP/VSP), levels=[1])
plt.contourf(rivec, svec, np.transpose(maskLSP*3), np.linspace(0,cl, 10), cmap='Greys')
plt.contour(rivec, svec, np.transpose(LSP/VSP), levels=[1], colors='b')
#plt.pcolormesh(rivec, svec, np.transpose(maskLSP*3),  cmap='Greys', shading='gouraud')

plt.contourf(rivec, svec, np.transpose(maskVSP*2), np.linspace(0,cl, 10), cmap='Greys')
#plt.contour(rivec, svec, np.transpose(maskLSP*mask), levels=[1], colors ='b')
plt.contour(rivec, svec, np.transpose(VSHS*maskPV), levels=[1], colors='b', linestyles='dashed')
plt.contour(rivec, svec, np.transpose(delta*maskPPV), levels=[1, 2], colors='k')
#plt.contour(rivec, svec, np.transpose(KET), levels=[0])
#plt.contour(rivec, svec, np.transpose(maskVSP*mask), levels=[1], color = 'b')
#%%
plt.figure()
plt.contourf(rivec, svec, np.transpose(VSP/KET), np.linspace(0, 2, 10))
plt.colorbar()
plt.contour(rivec, svec, np.transpose(PV), levels=[0], colors='r', linestyles='dashed') # Contour PV = 0 Line
plt.contour(rivec, svec, np.transpose(CI), levels=[1], colors='g', linestyles='dashed') # Contour the zero absolute vertical vorticity line

#%%
ax[2].contour(rivec, svec, np.transpose(delta), levels=[1], colors='k')
ax[2].contour(rivec, svec, np.transpose(maskdom*mask), levels=[1], colors='b')

ax[2].contour(rivec, svec, np.transpose(maskGR), levels=[1], colors='b')



plt.contourf(Ri, S, maskSI*4, np.linspace(0, cl, 10), cmap='Greys')

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
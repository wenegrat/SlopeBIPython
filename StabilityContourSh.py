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
directoryname = "../SlopeAngleShVar/"
directory = os.fsencode(directoryname)

ntht = 3
nll = 128

counter = 0
shvec = np.zeros(ntht)
rivec = np.zeros(ntht)

maxgr = np.zeros(ntht)
delta = np.zeros(ntht)
gr = np.zeros((ntht, nll), dtype=np.float64)    
grn = np.zeros((ntht, nll), dtype=np.float64)
ln = np.zeros((ntht, nll), dtype=np.float64)

plt.figure()
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    print(filename)
    if filename.endswith(".npz"): 
        a = np.load(directoryname+filename);
#        plt.semilogx(a['ll'], a['gr'])
        shvec[counter] = a['Sh']
        rivec[counter] = a['Ri']
        gr[counter, :] = a['gr']#[:,-1]
        grn[counter,:] = a['gr']/(a['f'])*np.sqrt(a['Ri'])
        ln[counter, :] = a['ll']*np.sqrt(a['Bz'][-1])*a['H']/a['f']
        ln[counter, :] = a['ll']*(a['Vz'][-1])*a['H']/a['f']
        maxgr[counter] = np.max(gr[counter,:])
        delta[counter] = a['Bz'][-1]/(a['f']*a['Vz'][-1]*np.cos(a['tht']))*a['tht']
#        grn[counter,:] = (gr[counter,:]/a['f'])*(rivec[counter]**(1/2))
        counter = counter + 1
#        plt.plot(a['ll'], gr[counter-1,:]*np.sqrt(a['Bz'][-1])/(a['f']*a['Vz'][-1]))
        continue
    else:
        continue
idx = np.argsort(shvec)
shvec = shvec[idx]
rivec = rivec[idx]
maxgr = maxgr[idx]
delta = delta[idx]
gr = gr[idx,:]
ln = ln[idx,:]
grn = grn[idx,:]
#grn = gr/a['f']

grnd = np.zeros(grn.shape)
#grnd[1:-1,:] = grn[1:-1,:]-grn[0:-2,:]
#grn[grnd>0.2] = np.nan

grnd[:, 1:-1] = grn[:, 1:-1]-grn[:,0:-2]
grn[np.abs(grnd)>0.05] = np.nan
#grn[grn==0] = np.nan
##valid_mask = ~np.isnan(grn)
##coords = np.array(np.nonzero(valid_mask)).T
##values = grn[valid_mask]
##it = interpolate.LinearNDInterpolator(coords, values)
##grn = it(list(np.ndindex(grn.shape))).reshape(grn.shape)
#
array = np.ma.masked_invalid(grn)
ll, tt = np.meshgrid(a['ll'], rivec)
l1 = ll[~array.mask]
t1 = tt[~array.mask]
newg = array[~array.mask]
grn = interpolate.griddata((l1, t1), newg.ravel(), (ll, tt))


#
#%%
# Make plot comparing Ri=1 with Stone solution
#ln = a['ll']/(a['f']/(a['V'][-1]))
k = np.linspace(0, 3, 100)
#ln = a['ll']*np.sqrt(a['Bz'][-1])*a['H']/a['f']
rind = 0
#Ri = shvec[rind]
Ri = rivec[rind]
#Ri = .1/shvec[rind]
stgr = 1/(2*3**(1/2))*(k - 2/15*k**3*(Ri + 1)) *np.sqrt(Ri)
stgr1 = 1/(2*3**(1/2))*(k - 2/15*k**3*(rivec[1] + 1))*np.sqrt(rivec[1])
stgr2 = 1/(2*3**(1/2))*(k - 2/15*k**3*(rivec[2] + 1))*np.sqrt(rivec[2])


rind = 2
Ri = rivec[rind]
delta = Ri*shvec[rind]
deltat=delta
km = np.sqrt(Ri)*k
stf = (np.max(stgr)-np.max(stgr1))/np.max(stgr)
mechoso = np.sqrt( (km-(1+deltat)*np.tanh(km))/np.tanh(km)*(1+delta)
         - 1/4*( (deltat -delta)/np.tanh(km) - km)**(2) + 0j)
    
    
plt.figure(figsize=(8, 4.5))
plt.plot(ln[0,:], grn[0,:])
plt.plot(ln[1,:], grn[1,:])
plt.plot(ln[2,:], grn[2,:])



plt.gca().set_prop_cycle(None)
plt.plot(k, stgr, linestyle='dashed')
plt.plot(k, stgr1, linestyle='dashed')

plt.plot(k, stgr2, linestyle='dashed')

plt.plot(k, mechoso)
#plt.plot(k, grtest/a['f'])
plt.ylim((0, .35))
plt.grid(linestyle='--', alpha = 0.5)

deltas = rivec*shvec
Sburg = deltas**2/rivec
#plt.title('SH: ' + str(shvec[rind]) + '  Ri: ' + str(Ri) + '  Slope Burger: ' + str((shvec[rind]**2*Ri)) + '   delta: '+str(Ri*shvec[rind]))
#plt.legend(['$\delta = S^{1/2} =$ ' + str(shvec[0]), '$\delta = S^{1/2} =$ '+str(shvec[1])])
plt.legend(['$S = ' + str(Sburg[0]) +'$ $S_H = '+str(shvec[0])+'$',
        '$S = ' + str(Sburg[1]) +'$ $S_H = '+str(shvec[1])+'$',
        '$S = ' + str(Sburg[2]) +'$ $S_H = '+str(shvec[2])+'$'])
plt.legend(['$\delta/Ri = '+str(shvec[0])+'$',
       '$\delta/Ri = '+str(shvec[1])+'$',
        '$\delta/Ri = '+str(shvec[2])+'$'])

#plt.figure(figsize=(10, 6))
#plt.plot(rivec, maxgr/a['f'])
#plt.ylim((-0.1, 0.1))
plt.xlim((0, 3))
plt.xlabel('$l\'$', fontsize=20)
#plt.ylabel('$\\omega Ri^{1/2} f^{-1}$', fontsize=20)
plt.ylabel('$\\omega/f$', fontsize=20)

plt.tight_layout()

#plt.savefig('/home/jacob/Dropbox/Slope BI/Slope BI Manuscript/SHDependenceRiF.eps', format='eps', dpi=1000)

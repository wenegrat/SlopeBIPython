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
import h5py
from pylab import *
import scipy.integrate as integrate
from cmocean import cm as cmo
import matplotlib.gridspec as gridspec
import matplotlib.ticker 
import scipy.stats as stats

plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 18})
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

# LOAD
a = np.load('/home/jacob/dedalus/NLSIM/StabilityData_1.5_day1.5.npz');
z = a['z']
#U = a['U']
V = a['V']
B = a['B']
Bz = a['Bz']
Vz = a['Vz']
tht = a['tht']
ll = a['ll']
gr = a['gr']
N = a['N']

#%% MAKE IDEALIZED GROWTH RATE PLOT (FIG 3)
plt.rcParams['text.usetex'] = True

def tickfun(X):
    Y = 1/X/1000
    return ['%i' % ceil(z) for z in Y]


fs = 20
plt.rc('xtick', labelsize=fs)
plt.rc('ytick', labelsize=fs)

fig = plt.figure(figsize=(12,5))
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()

ax1.semilogx(a['ll']/(2*np.pi), gr/a['f'], linewidth=2, marker='x')
ax1.set_xlabel('Along-slope wavenumber [m$^{-1}$]', fontsize = fs)
ax1.set_ylabel('Growth rate', fontsize=fs)
ax1.set_ylim((0, .25))
#ax1.set_xlim((1e-4, 1e-2))
#ax1.set_xlim((2e-5, 1.5e-3))
plt.grid(linestyle='--', alpha = 0.5)
#ax1.axvline(x=1/(6.5e3))
ax1.axvline(x=1/(5.5e3))
ax1.axvline(x=5/(32e3))
ax1.axvline(x=6/(32e3))


ax1.legend(fontsize=fs)
ax1.grid(linestyle='--', alpha = 0.5, which='Both')

newticks = np.array([2*np.pi/50e3, 2*np.pi/10e3, 2*np.pi/1e3])
newticks = np.array([1/50e3, 1/10e3, 1e-3])
ax2.set_xscale('log')

ax2.set_xticks(newticks)
ax2.set_xlim(ax1.get_xlim())

ax2.set_xticklabels(tickfun(newticks))
ax2.set_xlabel('Wavelength [km]', labelpad=5, fontsize=fs)
ax2.grid(False)
z = a['z']    

#fig.savefig('/home/jacob/Dropbox/Slope BI/Slope BI Manuscript/Revision 1/NLSim_Gr_Ri1.5.pdf', bbox_inches='tight')

#%% LOAD NL SIM SNAPS
filename = '/home/jacob/dedalus/NLSIM/snapshots/snapshots_s1.h5'
f = h5py.File(filename, 'r')

# List all groups
print("Keys: %s" % f.keys())
a_group_key = list(f.keys())[0]

# Get the data
data = list(f['tasks'])

b25 = f['tasks']['b 25']
b50 = f['tasks']['b 50']
nt = b25.shape[0]
nx = b25.shape[1]
ny = b25.shape[2]

bplane = f['tasks']['b plane']
u = f['tasks']['u']
v = f['tasks']['v']
w = f['tasks']['w']
b = f['tasks']['b']
ke = f['tasks']['ke']
mke = f['tasks']['mke']
#bp = f['tasks']['mean byncy prdctn']

x = u.dims[1][0][:]
y = u.dims[2][0][:]
z = u.dims[3][0][:]
time = u.dims[0][0][:]

eke = f['tasks']['eke']
eke = eke[:,:,:,:]*(x[-1])**2

wpbp = f['tasks']['wpbp']
sp = f['tasks']['sp'] #slope normal pert shear
diss = f['tasks']['vertical diss']
hypv = f['tasks']['hyper diss']
#mdiss = f['tasks']['mean vert diss']
#tsp = f['tasks']['tot shear prod']
#msp = f['tasks']['mf shear prod'] #slope normal prod mean flow
#lsp = f['tasks']['lat shear prod 2'] # lat shear produc 
#hypv = f['tasks']['hypv'] # only applies to pert quants
#diss = f['tasks']['dssptn'] #slope normal diss of pert quants
#fdiss = f['tasks']['f dssptn'] #slope normal diss of pert quants
#
#mfdiss = f['tasks']['mf dssptn'] # slope normal diss of background

V = f['tasks']['V']
Bz = f['tasks']['Bz']

vtot = v[:,:,:,:] + V[:,:,:,:] + 0.1
eked = 0.5*(np.mean(np.mean(u,axis=1), axis=1)**2 + np.mean(np.mean(v ,axis=1), axis=1)**2)
eked =integrate.trapz(eked, x=z, axis=-1)*x[-1]*y[-1]

#eketo = np.gradient(eke[:,0,0,0])/7200
eket = np.gradient(eke[:,0,0,0])/np.gradient(time)

#eket[1:] = (eke[1:,0,0,0] - eke[0:-1,0,0,0])/21600

nr = gr.size
gr[gr>a['f']] = 0
grt = np.float64(np.zeros((nr, time.size)))
for i in range(0, nr):
    grt[i,:] = np.exp(2*gr[i]*time)
stint1 = integrate.trapz(grt,x=a['ll'], axis=0)


#%% CALCULATE ALPHA PARAMETER

Ntot = np.sqrt(N**2-np.abs(Bz[0,0,0,:]*np.cos(tht))) #right way
#Ntot = N-np.sqrt(np.abs(Bz[0,0,0,:]*np.cos(tht))) #wrong way

#Ntot = np.sqrt(N**2 - 1e-4*0.1/100/np.sin(tht))
Sburger = Ntot**2/1e-8*np.tan(tht)**2
alpha = Ntot/1e-4*np.tan(tht)*np.sqrt(1.5)
alpha = np.sqrt(Sburger*1.5)
#%%
#ubar = np.mean(np.mean(u, axis=1), axis=1)
#dubar = np.gradient(ubar,  21600, axis=0)
#vbar = np.mean(np.mean(v, axis=1), axis=1)
#dvbar = np.gradient(vbar, 21600, axis=0)
#uprime = 0*u[:,:,:,:]
#vprime = 0*u[:,:,:,:]
#nterm = 0*u[:,:,:,:]
#for i in range(0, nx):
#    for j in range(0, ny):
#        uprime[:,i,j,:] = u[:,i,j,:] - ubar
#        vprime[:,i,j,:] = v[:,i,j,:] - vbar
#        nterm[:,i,j,:] = uprime[:,i,j,:]*dubar + vprime[:,i,j,:]*dvbar
#    
#%% Replicate to slope
nt, nx, ny, nz = V.shape
zhat = np.zeros((nx, nz))
xhat = np.zeros((nx, nz))
Bf = np.nan*np.zeros((nx,nz))
clm = [0, 0.12]
clm = [-0.01, 0.01]
cmap = 'viridis'
for i in range(0, nx):
    zhat[i,:] = (z + x[i]*np.tan(tht))
    xhat[i,:] = (x[i] - z*np.sin(tht))/1000
    Bf[i,1:] = integrate.cumtrapz(Bz[0,i,1,:]+(3.4e-3)**2*np.cos(tht), z) +(3.4e-3)**2*np.sin(tht)*x[i]

fig, ax = plt.subplots(2,1,sharex=False, figsize=(8,10)) 
a0 = ax[0].contourf(xhat, zhat, (V[0,:,1,:]+0.1), np.linspace(0, clm[1], 13), cmap=cmap)
ax[0].contour(xhat, zhat, Bf, 20, colors='0.5')
ts = 78
ts = range(60, 79)
a1 = ax[1].pcolor(xhat, zhat, np.mean(u[ts,:,1,:], axis=0), clim=clm, cmap=cmap, vmin=clm[0], vmax=clm[1])
ax[1].contour(xhat, zhat, np.mean(Bf + bplane[ts,:,0,:], axis=0), 20, colors='0.5')
plt.colorbar(a1,ax=ax[1], label='Along-slope velocity, m/s')
plt.colorbar(a0, ax=ax[0], label='Along-slope velocity, m/s')
ax[0].set_ylim((0, 500))
ax[1].set_ylim((0, 500))

ax[0].set_ylabel('z (m)')
ax[0].set_xlabel('km')
f = lambda x,pos: str(np.ceil(x)).rstrip('0').rstrip('.')

ax[0].set_xticks((np.linspace(x[0], x[-1], 3))/1000)
ax[0].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(f))
for c in a0.collections:
    c.set_edgecolor("face")

#fig.savefig('/home/jacob/Dropbox/Slope BI/Slope BI Manuscript/Revision 1/NLSim_Basic_Ri1.5.pdf', bbox_inches='tight')
#fig.savefig('/home/jacob/Dropbox/Presentations/OS 2018 Slope Presentation/Working Files/Figures/NLSim_Basic.pdf', bbox_inches='tight')
#%%
plt.figure(figsize=(7,4))
a0=plt.contourf(xhat, zhat, (V[0,:,1,:]+0.1), np.linspace(0, clm[1], 1*12+1), cmap=cmap)
plt.contour(xhat, zhat, Bf, 25, colors='0.5')
plt.colorbar(a0, label='Along-slope velocity, m/s')
plt.ylim((0,500))
plt.ylabel(r'$\hat{z}$ (m)')
plt.xlabel(r'$\hat{x}$ (km)')

for c in a0.collections:
    c.set_edgecolor("face")
#plt.savefig('/home/jacob/Dropbox/Slope BI/Slope BI Manuscript/Revision 1/NLSim_Basic.pdf', bbox_inches='tight')


#%% 3x Plan View Plots
nc = 75

t0 = 16
t1 = 37
t1 = 40

t2 = nt-1
t2 = 63
cm = cmo.ice
cm2 = cmo.balance
#cm2 = 'bwr'
bcl = np.linspace(0, 3.5e-3, nc)
wcl = np.linspace(-200, 200, nc)
fig, ax = plt.subplots(2, 4, sharey=False,sharex=False,figsize=(9*61/60, 6.0), gridspec_kw={'width_ratios':[20, 20, 20,1]})
a0 = ax[0, 0].contourf(x/1000,  y/1000, np.transpose(b50[t0,:,:,0])+(3.4e-3)**2*np.sin(tht)*x,bcl, extend='both', cmap=cm)
a1 = ax[0, 1].contourf(x/1000,  y/1000, np.transpose(b50[t1,:,:,0])+(3.4e-3)**2*np.sin(tht)*x,bcl, extend='both', cmap=cm)
a2 = ax[0, 2].contourf(x/1000,  y/1000, np.transpose(b50[t2,:,:,0])+(3.4e-3)**2*np.sin(tht)*x,bcl, extend='both', cmap=cm)
a3 = ax[1, 0].contourf(x/1000,  y/1000, np.transpose(w[t0,:,:,36])*86400, wcl, extend='both', cmap=cm2)
a4 = ax[1, 1].contourf(x/1000,  y/1000, np.transpose(w[t1,:,:,36])*86400, wcl, extend='both', cmap=cm2)
a5 = ax[1, 2].contourf(x/1000,  y/1000, np.transpose(w[t2,:,:,36])*86400,wcl, extend='both', cmap=cm2)

for c in a0.collections:
    c.set_edgecolor("face")
    
for c in a1.collections:
    c.set_edgecolor("face")
    
for c in a2.collections:
    c.set_edgecolor("face")
    
for c in a3.collections:
    c.set_edgecolor("face")
    
for c in a4.collections:
    c.set_edgecolor("face")
    
for c in a5.collections:
    c.set_edgecolor("face")
    
#a1 = ax[0, 0].contour(x/1000,  y/1000, np.transpose(b50[t1,:,:,0])+(3.4e-3)**2*np.sin(tht)*x,30, colors='0.5')
ax[1,3].axes.get_yaxis().set_visible(False)
ax[1,3].axes.get_xaxis().set_visible(False)
ax[1,3].spines['top'].set_visible(False)
ax[1,3].spines['left'].set_visible(False)
ax[1,3].spines['right'].set_visible(False)
ax[1,3].spines['bottom'].set_visible(False)
ax[0,3].axes.get_yaxis().set_visible(False)
ax[0,3].axes.get_xaxis().set_visible(False)
ax[0,3].spines['top'].set_visible(False)
ax[0,3].spines['left'].set_visible(False)
ax[0,3].spines['right'].set_visible(False)
ax[0,3].spines['bottom'].set_visible(False)
#plt.colorbar(a1, ax=ax[0,0])
#cbaxes = fig.add_axes([0.9, 0.5, 0.03, 0.5]) 
#cb = plt.colorbar(ax1, cax = cbaxes)  
cb1 = plt.colorbar(a2, ax=ax[0,3], ticks=[bcl[0], bcl[-1]], fraction=1)
cb1.set_label('Buoyancy \n m/s^2', labelpad=-2)
#plt.colorbar(a3, ax=ax[1,0])
cb2 = plt.colorbar(a4, ax=ax[1,3], ticks=[wcl[0], 0, wcl[-1]],fraction=1)
cb2.set_label('Slope normal velocity \n m/day', labelpad=6)
#cb = plt.colorbar(a1, ax=ax[0,0])
#cb.set_alpha(0)
ax[0,0].set_aspect('equal')
ax[0,1].set_aspect('equal')
ax[0,2].set_aspect('equal')
ax[1,0].set_aspect('equal')
ax[1,1].set_aspect('equal')
ax[1,2].set_aspect('equal')

ax[0,0].set_title('Day: ' + str(np.floor(time[t0]/86400)))
ax[0,1].set_title('Day: ' + str(np.floor(time[t1]/86400)))
ax[0,2].set_title('Day: ' + str(np.ceil(time[t2]/86400)))
#ax[0,0].set_xlabel('Across-slope, km')
#ax[0,1].set_xlabel('Across-slope, km')
ax[1,0].set_xlabel('Across-slope, km')
ax[1,1].set_xlabel('Across-slope, km')
ax[1,2].set_xlabel('Across-slope, km')
ax[0,0].set_ylabel('Along-slope, km')
#ax[0,1].set_ylabel('Along-slope, km')
ax[1,0].set_ylabel('Along-slope, km')
#ax[1,1].set_ylabel('Along-slope, km')

#0
f = lambda x,pos: str(np.ceil(x)).rstrip('0').rstrip('.')
ax[0,0].set_xticks((np.linspace(x[0], x[-1], 3)/1000))
ax[0,0].set_xticklabels([])
ax[0,0].set_yticks((np.linspace(x[0], x[-1], 3)/1000))
ax[0,0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(f))
#ax[0,0].set_yticklabels(np.ceil(np.linspace(x[0], x[-1], 3)/1000))

#1
ax[0,1].set_xticks((np.linspace(x[0], x[-1], 3)/1000))
ax[0,1].set_xticklabels([])
ax[0,1].set_yticks((np.linspace(x[0], x[-1], 3)/1000))
ax[0,1].set_yticklabels([])

#1
ax[0,2].set_xticks((np.linspace(x[0], x[-1], 3)/1000))
ax[0,2].set_xticklabels([])
ax[0,2].set_yticks((np.linspace(x[0], x[-1], 3)/1000))
ax[0,2].set_yticklabels([])
#2
ax[1,0].set_xticks((np.linspace(x[0], x[-1], 3)/1000))
#ax[1,0].set_xticklabels(np.ceil(np.linspace(x[0], x[-1], 3)/1000))
ax[1,0].set_yticks((np.linspace(x[0], x[-1], 3)/1000))
#ax[1,0].set_yticklabels(np.ceil(np.linspace(x[0], x[-1], 3)/1000))
ax[1,0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(f))
ax[1,0].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(f))

#3
ax[1,1].set_xticks((np.linspace(x[0], x[-1], 3)/1000))
ax[1,1].set_yticklabels([])
ax[1,1].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(f))

#3
ax[1,2].set_xticks((np.linspace(x[0], x[-1], 3)/1000))
ax[1,2].set_yticklabels([])
ax[1,2].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(f))


#fig.savefig('/home/jacob/Dropbox/Presentations/OS 2018 Slope Presentation/Working Files/Figures/NLSim.pdf', bbox_inches='tight')
#fig.savefig('/home/jacob/Dropbox/Slope BI/Slope BI Manuscript/Revision 1/NLSim_Overview.pdf', bbox_inches='tight')

#ax[1,1].set_xticklabels(np.ceil(np.linspace(x[0], x[-1], 3)/1000))
##%% Plan View Plots
#nc = 75
#
#t1 = 37
#t1 = 40
#t1 = 16
#t2 = nt-1
##t2 = 63
#cm = cmo.ice
#cm2 = cmo.balance
##cm2 = 'bwr'
#bcl = np.linspace(0, 3.5e-3, nc)
#wcl = np.linspace(-200, 200, nc)
#fig, ax = plt.subplots(2, 3, sharey=False,sharex=False,figsize=(9*41/40, 9.0), gridspec_kw={'width_ratios':[20,20,1]})
#a1 = ax[0, 0].contourf(x/1000,  y/1000, np.transpose(b50[t1,:,:,0])+(3.4e-3)**2*np.sin(tht)*x,bcl, extend='both', cmap=cm)
#a2 = ax[0, 1].contourf(x/1000,  y/1000, np.transpose(b50[t2,:,:,0])+(3.4e-3)**2*np.sin(tht)*x,bcl, extend='both', cmap=cm)
#a3 = ax[1, 0].contourf(x/1000,  y/1000, np.transpose(w[t1,:,:,18])*86400, wcl, extend='both', cmap=cm2)
#a4 = ax[1, 1].contourf(x/1000,  y/1000, np.transpose(w[t2,:,:,18])*86400,wcl, extend='both', cmap=cm2)
#
#for c in a1.collections:
#    c.set_edgecolor("face")
#    
#for c in a2.collections:
#    c.set_edgecolor("face")
#    
#for c in a3.collections:
#    c.set_edgecolor("face")
#    
#for c in a4.collections:
#    c.set_edgecolor("face")
#    
##a1 = ax[0, 0].contour(x/1000,  y/1000, np.transpose(b50[t1,:,:,0])+(3.4e-3)**2*np.sin(tht)*x,30, colors='0.5')
#ax[1,2].axes.get_yaxis().set_visible(False)
#ax[1,2].axes.get_xaxis().set_visible(False)
#ax[1,2].spines['top'].set_visible(False)
#ax[1,2].spines['left'].set_visible(False)
#ax[1,2].spines['right'].set_visible(False)
#ax[1,2].spines['bottom'].set_visible(False)
#ax[0,2].axes.get_yaxis().set_visible(False)
#ax[0,2].axes.get_xaxis().set_visible(False)
#ax[0,2].spines['top'].set_visible(False)
#ax[0,2].spines['left'].set_visible(False)
#ax[0,2].spines['right'].set_visible(False)
#ax[0,2].spines['bottom'].set_visible(False)
##plt.colorbar(a1, ax=ax[0,0])
##cbaxes = fig.add_axes([0.9, 0.5, 0.03, 0.5]) 
##cb = plt.colorbar(ax1, cax = cbaxes)  
#cb1 = plt.colorbar(a2, ax=ax[0,2], ticks=[bcl[0], bcl[-1]], fraction=1)
#cb1.set_label('Buoyancy \n m/s^2', labelpad=-4)
##plt.colorbar(a3, ax=ax[1,0])
#plt.colorbar(a4, ax=ax[1,2], ticks=[wcl[0], 0, wcl[-1]], label='Slope normal velocity \n m/day',fraction=1, pad=0)
##cb = plt.colorbar(a1, ax=ax[0,0])
##cb.set_alpha(0)
#ax[0,0].set_aspect('equal')
#ax[0,1].set_aspect('equal')
#ax[1,0].set_aspect('equal')
#ax[1,1].set_aspect('equal')
#
#
#ax[0,0].set_title('Day: ' + str(np.floor(time[t1]/86400)))
#ax[0,1].set_title('Day: ' + str(np.ceil(time[t2]/86400)))
##ax[0,0].set_xlabel('Across-slope, km')
##ax[0,1].set_xlabel('Across-slope, km')
#ax[1,0].set_xlabel('Across-slope, km')
#ax[1,1].set_xlabel('Across-slope, km')
#ax[0,0].set_ylabel('Along-slope, km')
##ax[0,1].set_ylabel('Along-slope, km')
#ax[1,0].set_ylabel('Along-slope, km')
##ax[1,1].set_ylabel('Along-slope, km')
#
##0
#f = lambda x,pos: str(np.ceil(x)).rstrip('0').rstrip('.')
#ax[0,0].set_xticks((np.linspace(x[0], x[-1], 3)/1000))
#ax[0,0].set_xticklabels([])
#ax[0,0].set_yticks((np.linspace(x[0], x[-1], 3)/1000))
#ax[0,0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(f))
##ax[0,0].set_yticklabels(np.ceil(np.linspace(x[0], x[-1], 3)/1000))
#
##1
#ax[0,1].set_xticks((np.linspace(x[0], x[-1], 3)/1000))
#ax[0,1].set_xticklabels([])
#ax[0,1].set_yticks((np.linspace(x[0], x[-1], 3)/1000))
#ax[0,1].set_yticklabels([])
##2
#ax[1,0].set_xticks((np.linspace(x[0], x[-1], 3)/1000))
##ax[1,0].set_xticklabels(np.ceil(np.linspace(x[0], x[-1], 3)/1000))
#ax[1,0].set_yticks((np.linspace(x[0], x[-1], 3)/1000))
##ax[1,0].set_yticklabels(np.ceil(np.linspace(x[0], x[-1], 3)/1000))
#ax[1,0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(f))
#ax[1,0].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(f))
#
##3
#ax[1,1].set_xticks((np.linspace(x[0], x[-1], 3)/1000))
##ax[1,1].set_xticklabels(np.ceil(np.linspace(x[0], x[-1], 3)/1000))
#ax[1,1].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(f))
#
#ax[1,1].set_yticks((np.linspace(x[0], x[-1], 3)/1000))
#ax[1,1].set_yticklabels([])
#fig.subplots_adjust(wspace=0.2, hspace=0)
#
##ax[0,0].set_ylim((0, 200))
##plt.tight_layout()

#fig.savefig('/home/jacob/Dropbox/Slope BI/Slope BI Manuscript/Revision 1/NLSim.pdf', bbox_inches='tight')

#%% CALCULATE POWER SPECTRAL DENSITY AS FUNCTION OF WAVENUMBER

time_step = x[2]-x[1]

ubar = integrate.trapz(integrate.trapz(u,x=x, axis=1),x=y, axis=1)/(x[-1]*y[-1])
vbar = integrate.trapz(integrate.trapz(v,x=x, axis=1),x=y, axis=1)/(x[-1]*y[-1])

uprime = 0*np.zeros(u.shape)
vprime = 0*np.zeros(u.shape)
for i in range(0, nx):
    for j in range(0, ny):
        uprime[:,i,j,:] = u[:,i,j,:] - ubar
        vprime[:,i,j,:] = v[:,i,j,:] - vbar

#%%
tn = 6*1 # Set the timestep to use the EKE spectrum from
nf = nx*1 # Periodic domain, so should be nx

#Preallocate
psf = np.zeros((nf,))
psfu = np.zeros((nf, ))+1j*0
psfv = np.zeros((nf, ))+1j*0
pst = np.zeros((nz, nf))
pstu = np.zeros((nz, nf))+1j*0
pstv = np.zeros((nz, nf))+1j*0

for i in range(0,nx):# Loop over each x
    dke = np.zeros(( nx,))
    for j in range(0, nz): #loop over each y
        datau = uprime[tn, i , : , j]
        datau = datau-np.mean(datau)
        datav = vprime[tn,i, :, j]
        datav = datav-np.mean(datav)

        datake = (datau**2 + datav**2)/2 #Alternate way to calculate it.
#        datake = datake-np.mean(datake)
        
        #Take FFTs
#        pst[j,:] = np.abs(np.fft.fft(datake, n=nf))
        psu = (np.fft.fft(datau, n=nf))
        psv = (np.fft.fft(datav, n=nf))
        pst[j,:] = (np.abs(psu)**2 + np.abs(psv)**2)/2
        pstu[j,:] = psu
        pstv[j,:] = psv
    # Sum across all x and y
    psf[:] += integrate.trapz(pst, x=z, axis=0)
    psfu[:] += integrate.trapz(pstu, x=z, axis=0)
    psfv[:] += integrate.trapz(pstv, x=z, axis=0)

#psfu = (np.fft.fft(integrate.trapz(integrate.trapz(uprime[tn,:,:,:], x=z, axis=-1), x=x, axis=0), n=nf))    
#psfv = (np.fft.fft(integrate.trapz(integrate.trapz(vprime[tn,:,:,:], x=z, axis=-1), x=x, axis=0), n=nf))    

# Normalization
psf = psf*time_step
psfu = psfu*time_step
psfv = psfv*time_step

# Make and sort wavenumber array
freqs = np.fft.fftfreq(psf.size, time_step)
idx = np.argsort(freqs)
freqs = freqs[idx]
psf = psf[idx]
psfu = psfu[idx]
psfv = psfv[idx]
# Turn into 1-sided FFT
psf[freqs<0] = 0
psfu[freqs<0] =0 
psfv[freqs<0] = 0
psf = 2*np.abs(psf)
psfu = 2*np.abs(psfu)
psfv = 2*np.abs(psfv)
#psf = psf/integrate.trapz(psf, x=freqs, axis=0)
#psfu = psfu/integrate.trapz(np.abs(psfu)**2, x=freqs, axis=0)
#psfv = psfv/integrate.trapz(np.abs(psfv)**2, x=freqs, axis=0)

# Interpolate linear stability growth rates to FFT frequencies
# Factor of 2 comes from the squared FFT.
gri = np.interp(freqs, a['ll']/(2*np.pi)*2, gr, left=0, right=0)
freqsalt = np.linspace(0, 20, 21)/x[-1] #Alternate discretization.
freqsalt = freqs[:]*2 
#gri = np.interp(freqsalt, a['ll']/(2*np.pi)*2, gr, left=0, right=0)
#psf = np.interp(freqsalt, freqs, psf)
#psfu = np.interp(freqsalt, freqs, psfu)
#psfv = np.interp(freqsalt, freqs, psfv)

grt = np.float64(np.zeros((psf.size, time.size)))
grtu = np.float64(np.zeros((psf.size, time.size)))+1j*0
grtv = np.float64(np.zeros((psf.size, time.size)))+1j*0
for i in range(0, psf.size):
    grt[i,:] = np.exp(2*gri[i]*1*(time-time[tn]))*psf[i]
#    grtu[i,:] = (np.exp(gri[i]*0.95*(time-time[tn]))*psfu[i])
#    grtv[i,:] = ( np.exp(gri[i]*0.95*(time-time[tn]))*psfv[i])
    
#    grt[i,:] = 0.5*(grtu[i,:]*np.conj(grtu[i,:]) + grtv[i,:]*np.conj(grtv[i,:]))
#    if gri[i]==0:
#        grt[i,:]=0
#    if freqs[i]<0:
#        grt[i,:] = 0
    
#stint = integrate.trapz(grt,x=freqsalt, axis=0)

stint = np.sum(grt, axis=0)

#%%
tp= np.zeros((nt, nf))
grestimate = np.zeros((nf,))

for i in range(0, nt):
    dke = integrate.trapz((integrate.trapz(uprime[i, :,:,:], x=z, axis=-1))/2, x=x, axis=0)
#    dke = integrate.trapz((integrate.trapz(vprime[i, :,:,:], x=z, axis=-1))/2, x=x, axis=0)

    tp[i,:] = np.abs(np.fft.fft(dke, n=nf))
    tp[i,:] = tp[i,idx]

#tp[tp==0] = 1e-10
for j in range(0, nf):
    grestimate[j] = np.polyfit(time[30:40],np.log(tp[20:30,j]),  1)[0]
#%%
plt.figure()
plt.semilogx(freqs, grestimate[:])
plt.semilogx(freqs, gri)
#%%
freqs = freqsalt
grn = gr/a['f']
psfn = psf
plt.figure()
plt.semilogx(freqs, freqs*psfn/np.max(psfn))
plt.semilogx(freqs, freqs*np.abs(psfu**2 + psfv**2)/np.max(np.abs(psfu**2+psfv**2)), marker='x')
plt.semilogx(freqs, freqs*np.abs(psfv)/np.max(np.abs(psfv)), marker='x')

plt.semilogx(freqs, freqs*gri/np.max(gri), label='Linear')
#plt.semilogx(a['ll']/(2*np.pi), gr)
plt.axvline(x=6/(x[-1]))
plt.legend()

#%%
ekealt = 0.5*integrate.trapz(integrate.trapz(integrate.trapz(uprime[:,:,:,:], x=z[:], axis=-1), x=x, axis=1)**2 + integrate.trapz(integrate.trapz( vprime[:,:,:,:], x=z[:], axis=-1), x=x, axis=1)**2, x=y, axis=1)
#upb = integrate.trapz(integrate.trapz(integrate.trapz((uprime[:,:,:,:]), x=z[:], axis=-1), x=x, axis=1), x=y, axis=1)**2
#vpb = integrate.trapz(integrate.trapz(integrate.trapz((vprime[:,:,:,:]), x=z[:], axis=-1), x=x, axis=1), x=y, axis=1)**2

#ekealt = 0.5*(upb+vpb)

#%%
plt.figure()
plt.semilogy(time/86400, eke[:,0,0,0])
plt.semilogy(time/86400, ekealt/1000, linestyle='dashed')

#%%
#stint = integrate.trapz(np.exp(2*gr[10:]), x=a['ll'][10:])
#tn = 12
#e1 = np.exp(np.max(gr)*2*time)/np.exp(np.max(gr)*2*time[tn])*eke[tn,0,0,0]
e1 = np.exp(np.max(gr)*2*1*(time-time[tn]))*eke[tn,0,0,0]
e1 = np.exp( np.interp(5/x[-1], a['ll']/(2*np.pi), gr)*2*(time-time[tn]))*eke[tn,0,0,0]/20
#e2 = np.exp( np.interp(6/32e3, a['ll']/(2*np.pi), gr)*2*(time-time[tn]))*eke[tn,0,0,0]/15

e2 = stint1/stint1[tn]*eke[tn,0,0,0]/10
e3 = stint/stint[tn]*eke[tn,0,0,0]*1

#e3 = stint*eke[tn,0,0,0]
#e3 = stint*time_step/nx
e3 = stint*time_step/nx
fig, ax = plt.subplots(2,1,figsize = (8,8), sharex=True)
#plt.figure(figsize=(5,4))
ax[0].semilogy(time/86400, eke[:,0,0,0], linewidth=3, label='EKE')
#ax[0].semilogy(time[:]/86400, ekealt[:]/40000, linewidth=3, label='EKE')

#ax[0].semilogy(time/86400, ekealt[:]/20000, linewidth=3, label='EKE')

#ax[0].semilogy(time/86400, eked[:], linewidth=3, label='EKE')

#ax[0].semilogy(time/86400, eke1[:,0,0,0]/2)

#ax[0].semilogy(time/86400, e1)
#ax[0].semilogy(time/86400, np.exp(np.max(gr)*1.65*time)*1e-11)

#ax[0].semilogy(time/86400, e2)
ax[0].semilogy(time/86400, e3, linestyle='dashed', linewidth=2, label='Linear theory')
ax[0].legend()
#ax[0].axvline(x=tn/4)
wpbpi = integrate.trapz(wpbp[:,0,0,:], x=z, axis=-1)/z[-1]
spi = integrate.trapz(sp[:,0,0,:], x=z, axis=-1)/z[-1]
dissi= integrate.trapz(diss[:,0,0,0:]+hypv[:,0,0,0:], x=z[0:], axis=-1)/z[-1]
#mdissi= integrate.trapz(mdiss[:,0,0,0:], x=z[0:], axis=-1)/z[-1]
sum = wpbpi + spi + dissi
ax[1].plot(time/86400, eket[:]/(x[-1]*y[-1]*z[-1]), label='EKE tendency', linewidth=3)
#ax[1].plot(time/86400, eketo[:]/(x[-1]*y[-1]*z[-1]), label='EKE tendency', linewidth=3)
ax[1].plot(time[:]/86400, wpbpi, linewidth=2, label='Buoyancy prod.')
ax[1].plot(time[:]/86400, spi, linewidth=2, label='Shear prod.')
ax[1].plot(time[:]/86400,dissi, linewidth=2, label='Dissipation')
#ax[1].plot(time/86400, sum[:], label='SUM', linewidth=3)
#ax[1].plot(time/86400,integrate.trapz(np.mean(np.mean(nterm, axis=1), axis=1), x=z, axis=-1)/z[-1], label='NT', linewidth=3)

#ax[1].plot(time[1:]/86400, integrate.cumtrapz(hypv[:,0,0,0], time))
#ax[1].set_yscale('symlog', linthreshy=1e-4)
ax[1].legend(loc=2)
#
#ax[1].plot(time/86400, eket[:], label='eke')
#ax[1].plot(time/86400, wpbpi)
#ax[1].plot(time/86400, sp[:,0,0,0])
#ax[1].plot(time/86400, diss[:,0,0,0])
#ax[1].plot(time/86400, hypv[:,0,0,0])
#ax[1].set_yscale('symlog', linthreshy=1e-12)

ax[0].set_yticks(np.logspace(-2, 10, 7))
ax[0].grid()
ax[1].grid()
ax[1].set_xlabel('Days')
#ax[0].set_title('Eddy kinetic energy')
ax[0].set_ylabel('m$^3$ m$^2$s$^{-2}$')
ax[1].set_ylabel('m$^2$s$^{-3}$')
ax[0].set_ylim((1e-2, 1e10))
ax[0].set_xlim((0, 20))
#ax[1].set_ylim((0,300))
ax[1].set_xlim((0,20))
ax[1].set_ylim((-1e-9, 2e-9))
#ax[1].set_title('Kinetic Energy Budget')
plt.tight_layout()

#fig.savefig('/home/jacob/Dropbox/Slope BI/Slope BI Manuscript/Revision 1/NLSim_Energetics.pdf', bbox_inches='tight')
#fig.savefig('/home/jacob/Dropbox/Presentations/OS 2018 Slope Presentation/Working Files/Figures/NLSim_EKE.pdf', bbox_inches='tight')
#fig.savefig('/home/jacob/Desktop/NLSim_Energetics2.pdf', bbox_inches='tight')

##%%
#plt.figure()
#semilogy(time/86400, eket)
#semilogy(time/86400, 2*np.max(gr)*np.exp(np.max(gr)*2*0.8*time)/1e12)
#%%
#plt.plot(time/86400, np.max(wpbp[:,0,0,:], axis=-1))
#%%
t1 = tn
plt.figure()
plt.plot(wpbp[t1,0,0,:], z)
plt.plot(sp[t1,0,0,:], z)
plt.plot(diss[t1,0,0,:]+hypv[t1,0,0,:], z)
#%% TEST GRADIENT
#tn = 24
#Bztotalraw = Bz[tn, 0, 0, :]  + N**2*np.cos(tht)
#Bztotal = Bztotalraw+np.gradient(b[tn,0,0,:], z)
#plt.figure()
#plt.plot(Bztotalraw, z)
#plt.plot(Bztotal, z)
#plt.ylim((0, 125))
#
#Vztotal = Vz + np.gradient(v[tn,0,0,:], z)
#plt.figure()
#plt.plot(Vz, z)
#plt.plot(Vztotal, z)
#plt.ylim((0,125))
#plt.xlim((-1e-2, 1e-2))
#%% w'b'
#wpbpi = integrate.trapz(wpbp[:,0,0,:], x=z, axis=-1)
#plt.figure()
#plt.semilogy(time/86400, wpbpi)
#plt.semilogy(time/86400, eket)
#plt.semilogy(time/86400, bp[:,0,0,0], label='bp')

##%%
#plt.figure()
#plt.scatter(np.log(eke[:,0,0,0]/500), np.log(eke1[:,0,0,0]/1000))
##plt.plot(np.linspace(-14, -7, 10), np.linspace(-14, -7, 10))
#plt.grid()






#%%
#plt.figure()
#plt.semilogx(freqs[idx], psf[idx], marker='x')
#plt.semilogx(a['ll']/(2*np.pi), gr)

##%% FANCY HEADER PLOT
#tb = np.transpose(b50[52,:,:,0])+(3.4e-3)**2*np.sin(tht)*x
#tbf = np.tile(tb, (2, 1))
#nc = 100
#bcl = np.linspace(0, 3.5e-3, nc)
#plt.figure(figsize=(8, 1))
#a1 = plt.contourf(np.transpose(tbf), bcl, extend='both', cmap=cm)
#for c in a1.collections:
#    c.set_edgecolor("face")
##axis('equal')
#plt.ylim((0, 64))
#plt.xlim((0, 256*2))
#axis('equal')
##plt.savefig('/home/jacob/Dropbox/Presentations/OS 2018 Slope Presentation/Working Files/Figures/Header.pdf', bbox_inches='tight')

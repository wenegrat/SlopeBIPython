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
a = np.load('/home/jacob/dedalus/NLSIM/StabilityData_1.5.npz');
z = a['z']
U = a['U']
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
filename = '/home/jacob/dedalus/NLSIM/snapshots/snapshots_s1_long.h5'
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
bp = f['tasks']['mean byncy prdctn']

x = u.dims[1][0][:]
y = u.dims[2][0][:]
z = u.dims[3][0][:]
time = u.dims[0][0][:]

eke = f['tasks']['eke']*x[-1]*y[-1]

wpbp = f['tasks']['wpbp']
sp = f['tasks']['shear prod'] #slope normal pert shear
msp = f['tasks']['mf shear prod'] #slope normal prod mean flow
lsp = f['tasks']['lat shear prod 2'] # lat shear produc 
hypv = f['tasks']['hypv'] # only applies to pert quants
diss = f['tasks']['dssptn'] #slope normal diss of pert quants
mfdiss = f['tasks']['mf dssptn'] # slope normal diss of background

V = f['tasks']['V']
Bz = f['tasks']['Bz']



eket = np.gradient(eke[:,0,0,0], 21600)


nr = gr.size
gr[gr>a['f']] = 0
grt = np.float64(np.zeros((nr, time.size)))
for i in range(0, nr):
    grt[i,:] = np.exp(2*gr[i]*time)
stint1 = integrate.trapz(grt,x=a['ll'], axis=0)
#%% Replicate to slope
nt, nx, ny, nz = V.shape
zhat = np.zeros((nx, nz))
xhat = np.zeros((nx, nz))
Bf = np.nan*np.zeros((nx,nz))
clm = [0, 0.12]
cmap = 'viridis'
for i in range(0, nx):
    zhat[i,:] = (z + x[i]*np.tan(tht))
    xhat[i,:] = (x[i] - z*np.sin(tht))/1000
    Bf[i,1:] = integrate.cumtrapz(Bz[0,i,1,:]+(3.4e-3)**2*np.cos(tht), z) +(3.4e-3)**2*np.sin(tht)*x[i]

fig, ax = plt.subplots(2,1,sharex=False, figsize=(8,10)) 
a0 = ax[0].contourf(xhat, zhat, (V[0,:,1,:]+0.1), np.linspace(0, clm[1], 13), cmap=cmap)
ax[0].contour(xhat, zhat, Bf, 20, colors='0.5')
ts = 78
a1 = ax[1].pcolor(xhat, zhat, (V[ts,:,1,:]+0.1+ np.mean(v[ts,:,:,:],axis=1)), clim=clm, cmap=cmap, vmin=clm[0], vmax=clm[1])
ax[1].contour(xhat, zhat, Bf + bplane[ts,:,0,:], 20, colors='0.5')
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
a3 = ax[1, 0].contourf(x/1000,  y/1000, np.transpose(w[t0,:,:,18])*86400, wcl, extend='both', cmap=cm2)
a4 = ax[1, 1].contourf(x/1000,  y/1000, np.transpose(w[t1,:,:,18])*86400, wcl, extend='both', cmap=cm2)
a5 = ax[1, 2].contourf(x/1000,  y/1000, np.transpose(w[t2,:,:,18])*86400,wcl, extend='both', cmap=cm2)

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

#%%
tn = 12
time_step = 500
psf = np.zeros((nx,))
grt = np.float64(np.zeros((nx, time.size)))
pst = np.zeros((nz, nx))

for i in range(0,nx):
    for j in range(0, nz):
        datau = u[tn,i,:,j]
        datau = datau-np.mean(datau)
        datav = v[tn,i,:,j]
        datav = datav-np.mean(datav)
        datake = datau**2 + datav**2
        datake = datake-np.mean(datake)
        psu = np.abs(np.fft.fft(datau))
        psv = np.abs(np.fft.fft(datav))
        pst[j,:] = ((psu)**2 + psv**2)
        pst[j,:] = np.abs(np.fft.fft(datake))
    psf[:] += integrate.trapz(pst, x=z, axis=0)
        
psf = psf/(nx*nz)     
   
freqs = np.fft.fftfreq(datau.size, time_step)
idx = np.argsort(freqs)
freqs = freqs[idx]
psf = psf[idx]
gri = np.interp(freqs, a['ll']/(2*np.pi), gr, left=0, right=0)
for i in range(0, nx):
    grt[i,:] = np.exp(2*gri[i]*(time))*psf[i]
#    if gri[i]==0:
#        grt[i,:] = 0
    
stint = integrate.trapz(grt,x=freqs, axis=0)


#%%
#stint = integrate.trapz(np.exp(2*gr[10:]), x=a['ll'][10:])
tn = 12
e1 = np.exp(np.max(gr)*2*time)/np.exp(np.max(gr)*2*time[tn])*eke[tn,0,0,0]
e2 = stint1/stint1[tn]*eke[tn,0,0,0]
e3 = stint/stint[tn]*eke[tn,0,0,0]
#e3 = stint*60
fig, ax = plt.subplots(2,1,figsize = (8,8), sharex=True)
#plt.figure(figsize=(5,4))
ax[0].semilogy(time/86400, eke[:,0,0,0], linewidth=3, label='EKE')
#ax[0].semilogy(time/86400, eke1[:,0,0,0]/2)

#ax[0].semilogy(time/864001, e1)
#ax[0].semilogy(time/86400, np.exp(np.max(gr)*1.65*time)*1e-11)

#ax[0].semilogy(time/86400, e2)
ax[0].semilogy(time/86400, e3, linestyle='dashed', linewidth=2, label='Linear theory')
ax[0].legend()

wpbpi = integrate.trapz(wpbp[:,0,0,:], x=z, axis=-1)/z[-1]
spi = integrate.trapz(sp[:,0,0,:]+0*lsp[:,0,0,:]+msp[:,0,0,:], x=z, axis=-1)/z[-1]
dissi= integrate.trapz(diss[:,0,0,:]+hypv[:,0,0,:]+mfdiss[:,0,0,:], x=z, axis=-1)/z[-1]
sum = wpbpi + spi+dissi 
ax[1].plot(time/86400, eke[:,0,0,0], label='EKE', linewidth=3)
ax[1].plot(time[1:]/86400, integrate.cumtrapz(wpbpi, time), linewidth=2, label='VBP')
ax[1].plot(time[1:]/86400, integrate.cumtrapz(spi, time), linewidth=2, label='SP')
ax[1].plot(time[1:]/86400, integrate.cumtrapz(dissi, time), linewidth=2, label='DISS')
ax[1].plot(time[1:]/86400, integrate.cumtrapz(sum[:], time), label='SUM')
ax[1].set_yscale('symlog', linthreshy=1e-4)
ax[1].set_yticks([-1e0, -1e-4, 0, 1e-4, 1e-0])
ax[1].legend()
#
#ax[1].plot(time/86400, eket[:], label='eke')
#ax[1].plot(time/86400, wpbpi)
#ax[1].plot(time/86400, sp[:,0,0,0])
#ax[1].plot(time/86400, diss[:,0,0,0])
#ax[1].plot(time/86400, hypv[:,0,0,0])
#ax[1].set_yscale('symlog', linthreshy=1e-12)


ax[0].grid()
ax[1].grid()
ax[1].set_xlabel('Days')
ax[0].set_title('Eddy kinetic energy')
ax[0].set_ylabel('m^2s^{-2}')
ax[0].set_ylim((1e-11, 1e1))
ax[0].set_xlim((0, 20))
#ax[1].set_ylim((0,300))
ax[1].set_xlim((0,20))

ax[1].set_title('Kinetic Energy Budget')
plt.tight_layout()


#%%
#stint = integrate.trapz(np.exp(2*gr[10:]), x=a['ll'][10:])
tn = 12
e1 = np.exp(np.max(gr)*2*time)/np.exp(np.max(gr)*2*time[tn])*eke[tn,0,0,0]
e2 = stint1/stint1[tn]*eke[tn,0,0,0]
e3 = stint/stint[tn]*eke[tn,0,0,0]
#e3 = stint*60
fig, ax = plt.subplots(2,1,figsize = (8,8), sharex=True)
#plt.figure(figsize=(5,4))
ax[0].semilogy(time/86400, eke[:,0,0,0], linewidth=3, label='EKE')
#ax[0].semilogy(time/86400, eke1[:,0,0,0]/2)

#ax[0].semilogy(time/864001, e1)
#ax[0].semilogy(time/86400, np.exp(np.max(gr)*1.65*time)*1e-11)

#ax[0].semilogy(time/86400, e2)
ax[0].semilogy(time/86400, e3, linestyle='dashed', linewidth=2, label='Linear theory')
ax[0].legend()

sum = wpbpi + spi + dissi
ax[1].plot(time/86400, eket[:]/(x[-1]*y[-1]*z[-1]), label='EKE tendency', linewidth=3)
ax[1].plot(time[:]/86400, wpbpi, linewidth=2, label='Buoyancy prod.')
ax[1].plot(time[:]/86400, spi, linewidth=2, label='Shear prod.')
ax[1].plot(time[:]/86400,dissi, linewidth=2, label='Dissipation')
#ax[1].plot(time/86400, sum[:], label='SUM', linewidth=3)

#ax[1].plot(time[1:]/86400, integrate.cumtrapz(hypv[:,0,0,0], time))
#ax[1].set_yscale('symlog', linthreshy=1e-4)
ax[1].legend()
#
#ax[1].plot(time/86400, eket[:], label='eke')
#ax[1].plot(time/86400, wpbpi)
#ax[1].plot(time/86400, sp[:,0,0,0])
#ax[1].plot(time/86400, diss[:,0,0,0])
#ax[1].plot(time/86400, hypv[:,0,0,0])
#ax[1].set_yscale('symlog', linthreshy=1e-12)


ax[0].grid()
ax[1].grid()
ax[1].set_xlabel('Days')
#ax[0].set_title('Eddy kinetic energy')
ax[0].set_ylabel('m^3\;m^2s^{-2}')
ax[1].set_ylabel('m^2s^{-3}')
ax[0].set_ylim((1e-2, 5e9))
ax[0].set_xlim((0, 20))
#ax[1].set_ylim((0,300))
ax[1].set_xlim((0,20))

#ax[1].set_title('Kinetic Energy Budget')
plt.tight_layout()

#fig.savefig('/home/jacob/Dropbox/Slope BI/Slope BI Manuscript/Revision 1/NLSim_EKE_Ri1.5.pdf', bbox_inches='tight')
#fig.savefig('/home/jacob/Dropbox/Presentations/OS 2018 Slope Presentation/Working Files/Figures/NLSim_EKE.pdf', bbox_inches='tight')

##%%
#plt.figure()
#semilogy(time/86400, eket)
#semilogy(time/86400, 2*np.max(gr)*np.exp(np.max(gr)*2*0.8*time)/1e12)
#%%
plt.plot(time/86400, np.max(wpbp[:,0,0,:], axis=-1))
#%%
plt.figure()
plt.plot(wpbp[t1,0,0,:], z)
plt.plot(sp[t1,0,0,:]+msp[t1,0,0,:], z)
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
wpbpi = integrate.trapz(wpbp[:,0,0,:], x=z, axis=-1)
plt.figure()
plt.semilogy(time/86400, wpbpi)
plt.semilogy(time/86400, eket)
plt.semilogy(time/86400, bp[:,0,0,0], label='bp')

#%%
plt.figure()
plt.scatter(np.log(eke[:,0,0,0]/500), np.log(eke1[:,0,0,0]/1000))
#plt.plot(np.linspace(-14, -7, 10), np.linspace(-14, -7, 10))
plt.grid()






#%%
plt.figure()
plt.semilogx(freqs[idx], psf[idx], marker='x')
plt.semilogx(a['ll']/(2*np.pi), gr)
#%%
tb = np.transpose(b50[52,:,:,0])+(3.4e-3)**2*np.sin(tht)*x
tbf = np.tile(tb, (2, 1))
nc = 100
bcl = np.linspace(0, 3.5e-3, nc)
plt.figure(figsize=(8, 1))
a1 = plt.contourf(np.transpose(tbf), bcl, extend='both', cmap=cm)
for c in a1.collections:
    c.set_edgecolor("face")
#axis('equal')
plt.ylim((0, 64))
plt.xlim((0, 256*2))
axis('equal')
#plt.savefig('/home/jacob/Dropbox/Presentations/OS 2018 Slope Presentation/Working Files/Figures/Header.pdf', bbox_inches='tight')

"""
Dedalus script for calculating the maximum growth rates in no-slip
Rayleigh Benard convection over a range of horizontal wavenumbers.

This script can be ran serially or in parallel, and produces a plot of the
highest growth rate found for each horizontal wavenumber.

To run using 4 processes, for instance, you could use:
    $ mpiexec -n 4 python3 rayleigh_benard.py

"""

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
plt.rc('text', usetex=True)
import dedalus.public as de
from mpi4py import MPI
CW = MPI.COMM_WORLD
from pylab import *
import os

import scipy.integrate as integrate

import logging
logger = logging.getLogger(__name__)


# Global parameters
directoryname = "/home/jacob/dedalus/SlopeAngleRi1/"

# Variable Parameters
#ln = 2.1# 0.25
#delta = -0.5

# Physical parameters
#f = 1e-4
#tht = 0
#Pr = 1
#H = 150
#H = 50
##BLH = 250
##BLH = 50
#Ri = 0
##RiBL = 1
##RiBL = 0.0
##Bzmag = 2.5e-5
##Shmag = np.sqrt(Bzmag/Ri)
#Shmag = .1/H
##Shmag = 0.1/BLH
#Bzmag = Ri*Shmag**2
#BzmagBL = RiBL*Shmag**2


import glob

nll = 192
nz = 256
#nz = 128
directoryname = "../GOTMEkmanCluster/"
directoryname = "../GOTMEkmanD/"
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
Vzs = np.nan*np.zeros((len(dirfiles), nz))
Bzs = np.nan*np.zeros((len(dirfiles), nz))

Nus = np.nan*np.zeros((len(dirfiles), nz))
Kappas = np.nan*np.zeros((len(dirfiles), nz))
Bs = np.nan*np.zeros((len(dirfiles), nz))
Bzf = np.nan*np.zeros((len(dirfiles), nz))
us = np.nan*np.zeros((len(dirfiles), nz))
vs = np.nan*np.zeros((len(dirfiles), nz))
bs = np.nan*np.zeros((len(dirfiles), nz))
ws = np.nan*np.zeros((len(dirfiles), nz))
sps = np.nan*np.zeros((len(dirfiles), nz))
bps = np.nan*np.zeros((len(dirfiles), nz))
diss = np.nan*np.zeros((len(dirfiles), nz))

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
            Vzs[counter,:] = a['Vz']
            Bzs[counter,:] = a['Bz']
            Kappas[counter,:]  = a['kap']
            Nus[counter,:]  = a['nu']   
            Bs[counter,:]  = a['B']            
            Bzf[counter,:] = a['Bz']
            us[counter,:] = a['u']
            vs[counter,:] = a['v']
            bs[counter,:] = a['b']
            ws[counter,:] = a['w']
            sps[counter,:] = a['SP']
            bps[counter,:] = a['BP']
            diss[counter,:] = a['DISS']
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



#%% SETUP EIGENVALUE PROBLEM
H = 150
tind = next(x[0] for x in enumerate(time*a['f']/(2*np.pi)) if x[1] > 10) -1

#l = ln*f/(H*np.sqrt(Bzmag))
l = 0.0040392358531509045
idx = np.argmax(grf[tind,:])
l = a['ll'][idx]
#l = 0.00022539339047347913 #deep mode
#l = 0.009
#l = 1e-6

#tht = delta/(Bzmag/(f*Shmag))
tht = 5e-3
tht = 0.01

# Grid Parameters
nz = 256
#nz = 128

# Create bases and domain
# Use COMM_SELF so keep calculations independent between processes
z_basis = de.Chebyshev('z', nz, interval=(0,H))
domain = de.Domain([z_basis], grid_dtype=np.complex128, comm=MPI.COMM_SELF)

z = domain.grid(0)

# Define Stability Analysis Parameters

kap = domain.new_field(name='kap')
kap['g'] = Kappas[tind,:]
nu = domain.new_field(name='nu')
nu['g'] = Nus[tind,:]
U = domain.new_field(name='U')
U['g'] = Us[tind,:]
Uz = domain.new_field(name='Uz')
Uz = U.differentiate(z_basis)
V = domain.new_field(name='V')
V['g'] = Vs[tind,:]
Vz = domain.new_field(name='Vz')
Vz['g'] = Vzs[tind,:]
Bz = domain.new_field(name='Bz')
B = domain.new_field(name='B')
B['g'] = Bs[tind,:]
Bz = B.differentiate(z_basis)
#V['g'] = Shmag*(z)
#Vz['g'] = Shmag*(z-z+1) #Note this assumes no horizotal variation (ie. won't work for the non-uniform case)
#Bt = np.zeros([nz])
#Bz['g'] = np.array(Bzmag*np.ones([nz]))
#zind = np.floor( next((x[0] for x in enumerate(z) if x[1]>BLH)))
#Bz['g'][0:zind] = BzmagBL

#tpoint = np.floor( next((x[0] for x in enumerate(z) if x[1]>BLH)))
#Bstr  = -0.5*(np.tanh((-z + z[tpoint])/40)+1)
#Bstr  = -0.5*(np.tanh((-z + z[tpoint])/5)+1)

#Bz['g'] = Bz['g']*10**(2*Bstr)
#Bz['g'] = Bz['g']-Bz['g'][0]    # hack to check zero stratification BBL
    
#Bt[1:nz] = integrate.cumtrapz(Bz['g'], z)
#B['g'] = Bt
# 2D Boussinesq hydrodynamics, with no-slip boundary conditions
# Use substitutions for x and t derivatives
#    bz = 1/(2*Ri*np.sin(tht)**2)*(f**2*np.cos(tht)+np.sqrt(f**4*np.cos(tht)**2 + 4*Bzmag*f**2*Ri*np.sin(tht)**2))
#    Shmag = -bz/(f*np.sin(tht))



problem = de.EVP(domain, variables=['u', 'v', 'w', 'b', 'p', 'uz', 'vz', 'wz',
        'bz'], eigenvalue='omg', tolerance = 1e-10)

problem.parameters['tht'] = tht
problem.parameters['U'] = U
problem.parameters['V'] = V
problem.parameters['B'] = B
problem.parameters['Uz'] = Uz
problem.parameters['Vz'] = Vz
problem.parameters['Bz'] = Bz
problem.parameters['N'] =3.5e-3
problem.parameters['f'] = 1e-4
problem.parameters['kap'] = kap
problem.parameters['nu'] = nu
problem.parameters['k'] = 0. # will be set in loop
problem.parameters['l'] = l # will be set in loop
problem.substitutions['dx(A)'] = "1j*k*A"
problem.substitutions['dy(A)'] = "1j*l*A"
problem.substitutions['dt(A)'] = "-1j*omg*A"
problem.add_equation(('dt(u) + U*dx(u) + V*dy(u) + w*Uz - f*v*cos(tht) + dx(p)'
        '- b*sin(tht) - (dz(nu)*uz'
        '+ nu*dz(uz)) = 0'))
problem.add_equation(('dt(v) + U*dx(v) + V*dy(v) + w*Vz + f*u*cos(tht)'
        '- f*w*sin(tht) + dy(p) - (dz(nu)*vz + nu*dz(vz)) = 0'))
problem.add_equation(('(dt(w) + U*dx(w) + V*dy(w)) + f*v*sin(tht) + dz(p)'
        '- b*cos(tht) - (dz(nu)*wz + nu*dz(wz)) = 0'))
problem.add_equation(('dt(b) + U*dx(b) + V*dy(b) + u*N**2*sin(tht)'
            '+ w*Bz - dz(kap)*bz - kap*dz(bz) = 0'))
problem.add_equation('dx(u) + dy(v) + wz = 0')
problem.add_equation('uz - dz(u) = 0')
problem.add_equation('vz - dz(v) = 0')
problem.add_equation('wz - dz(w) = 0')
problem.add_equation('bz - dz(b) = 0')
problem.add_bc('left(u) = 0')
problem.add_bc('left(v) = 0')
problem.add_bc('left(w) = 0')
problem.add_bc('left(bz) = 0')
problem.add_bc('right(uz) = 0')
problem.add_bc('right(vz) = 0')
problem.add_bc('right(w) = 0')
problem.add_bc('right(bz) = 0')

solver = problem.build_solver()
    

def sorted_eigen(k, l):

    """
    Solves eigenvalue problem and returns sorted eigenvalues and associated
    eigenvectors.
    """
    # solve problem
    solver.solve_dense(solver.pencils[0], rebuild_coeffs=True)

    # sort eigenvalues
    omg = solver.eigenvalues
    omg[np.isnan(omg)] = 0.
    omg[np.isinf(omg)] = 0.
    idx = np.argsort(omg.imag)

    return idx
idx = sorted_eigen(0., 0)
solver.set_state(idx[-1])
maxgr = solver.eigenvalues[idx[-1]].imag
# collect eigenvector
u = solver.state['u']
uz = solver.state['uz']
vz = solver.state['vz']
v = solver.state['v']
w = solver.state['w']
b = solver.state['b']

#%%
# shear production
Vx = -Vz['g']*np.sin(tht)

VSP = -2*np.real((np.conj(w['g'])*v['g']*Vz['g']*np.cos(tht) + np.conj(v['g'])*u['g']*np.sin(tht)*Vz['g']))/np.cos(tht)
VSP = VSP - 2*np.real((np.conj(w['g'])*u['g']*Uz['g']*np.cos(tht) + np.conj(u['g'])*u['g']*np.sin(tht)*Uz['g']))

Ux = -Uz['g']*np.sin(tht)
LSP = -2*np.real(-np.conj(w['g'])*v['g']*Vx*np.sin(tht) + np.conj(v['g'])*u['g']*Vx)
LSP = LSP - 2*np.real(-np.conj(w['g'])*u['g']*Ux*np.sin(tht) + np.conj(u['g'])*u['g']*Ux)

SP = -2*np.real(np.conj(w['g'])*(u['g']*Uz['g']+v['g']*Vz['g']))

# buoyancy production
BP = 2*np.real((u['g']*np.sin(tht)+w['g']*np.cos(tht))*np.conj(b['g']))
DISS = -2*np.real(nu['g']*(np.conj(uz['g'])*uz['g'] + np.conj(vz['g'])*vz['g']))

# SAVE TO FILE

#np.savez(name + '.npz', nz=nz, N=N, tht=tht, z=z, kap=kap['g'], Pr=Pr, U=U['g'],
#        V=V['g'], B=B['g'], u=u['g'], v=v['g'], w=w['g'], b=b['g'], ll=ll,
#        gr=gr, SP=SP, BP=BP)
#%%
# PLOTTING

# mean state

#fig, ax = plt.subplots(1, 3, sharey=True)
#
#ax[0].semilogx(kap['g'], z)
#ax[0].set_xlabel('mixing coefficient [m$^2$/s]', va='baseline')
#ax[0].set_ylabel('slope-normal coordinate [m]')
#ax[0].get_xaxis().set_label_coords(.5, -.12)
#
#ax[1].plot(U['g'].real, z)
#ax[1].plot(V['g'].real, z)
#ax[1].set_xlabel('mean flow [m/s]', va='baseline')
#ax[1].get_xaxis().set_label_coords(.5, -.12)
#
#ax[2].plot(N**2*np.cos(tht)*z + B['g'].real, z)
#ax[2].set_xlabel('mean buoyancy [m/s$^2$]', va='baseline')
#ax[2].get_xaxis().set_label_coords(.5, -.12)

#fig.savefig('fig/mean_state.pdf')

# energetics
#%%

#plt.savefig('fig/growth_rate.pdf')
#%%
#fs = 16
#plt.figure(figsize=(5, 5))
#plt.plot(BP/np.max(BP), z)
#plt.plot((VSP+LSP)/np.max(BP), z)
##plt.plot(LSP/np.max(BP), z)
#
#plt.xlabel('Kinetic Energy Tendency', fontsize=fs)
#plt.ylabel('slope-normal coordinate [m]', fontsize=fs)
#plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
#plt.legend(['Buoyancy Production', 'Shear Production'], frameon=False, fontsize=fs, loc=1)
#plt.tight_layout()
#plt.ylim((0, 1000))
#plt.grid(linestyle='--', alpha = 0.5)

#plt.xlim((-0.1, 1))
#plt.savefig('/home/jacob/Dropbox/Slope BI/Slope BI Manuscript/IdealizedEnergetics.eps', format='eps', dpi=1000, bbox_inches='tight')

#%%
## most unstable mode
#nc  = 40
#ly = np.linspace(0, 2*np.pi, nz)
#uvel = np.real(u['g'].reshape(nz, 1)* np.exp(1j*ly.reshape(1,nz)))
#vvel = np.real(v['g'].reshape(nz, 1)* np.exp(1j*ly.reshape(1,nz)))
#wvel = np.real(w['g'].reshape(nz, 1)* np.exp(1j*ly.reshape(1,nz)))
#maxu = np.max(uvel)
#
#uvel = uvel/maxu
#vvel = vvel/maxu
#wvel = wvel/maxu
#buoy = np.real(b['g'].reshape(nz, 1)* np.exp(1j*ly.reshape(1,nz)))
#buoy = buoy/np.max(buoy)
#
#fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 10))
## UVEL
#im = ax[0,0].contourf(ly, z, uvel, np.linspace(-1, 1, nc),vmin=-1, vmax=1, cmap='RdBu_r')
#cb = plt.colorbar(im, ax=ax[0,0])
#cb.set_ticks([-1, 0, 1])
#ax[0,0].set_title('Across-slope velocity', fontsize=fs)
#ax[0,0].grid(linestyle='--', alpha = 0.5)
#
## VVEL
#im = ax[0,1].contourf(ly, z, vvel, np.linspace(-0.6, 0.6, nc),vmin=-0.6, vmax=0.6, cmap='RdBu_r')
#cb = plt.colorbar(im, ax=ax[0,1])
#cb.set_ticks([-0.6, 0, 0.6])
#ax[0,1].set_title('Along-slope velocity', fontsize=fs)
#ax[0,1].grid(linestyle='--', alpha = 0.5)
#
## WVEL
#im = ax[1,0].contourf(ly, z, wvel, np.linspace(-0.15, 0.15, nc),vmin=-0.15, vmax=0.15, cmap='RdBu_r')
#cb = plt.colorbar(im, ax=ax[1,0])
#cb.set_ticks([-0.15, 0, 0.15])
#ax[1,0].set_title('Slope-normal velocity', fontsize=fs)
#ax[1,0].grid(linestyle='--', alpha = 0.5)
#
## BUOY
#im = ax[1,1].contourf(ly, z, buoy, np.linspace(-1, 1, nc),vmin=-1, vmax=1, cmap='RdBu_r')
#cb = plt.colorbar(im, ax=ax[1,1])
#ax[1,1].grid(linestyle='--', alpha = 0.5)
#cb.set_ticks([-1, 0, 1])
#ax[1,1].set_title('Buoyancy', fontsize=fs)
#ax[0,0].set_xticks([0, np.pi, 2*np.pi])
#ax[1,0].set_xlabel('Phase', fontsize=fs)
#ax[1,1].set_xlabel('Phase', fontsize=fs)
#ax[0,0].set_ylabel('Slope-normal coordinate [m]', fontsize=fs)
#ax[1,0].set_ylabel('Slope-normal coordinate [m]', fontsize=fs)
#
##labels = [item.get_text() for item in ax[1,1].get_xticklabels()]
#labels = ['0', '$\pi$', '$2\pi$']
#ax[1,1].set_xticklabels(labels)  
##plt.savefig('fig/modes.pdf', dpi=300)
#plt.tight_layout()
##plt.savefig('/home/jacob/Dropbox/Slope BI/Slope BI Manuscript/IdealizedPerturbations.pdf', format='pdf')
#
#plt.show()

#%%
cm = 'RdBu_r'
# most unstable mode
fs = 18

plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': fs})
plt.rcParams['contour.negative_linestyle'] = 'solid'

nc  = 16
#nc = 20
lw =0.5
ly = np.linspace(0, 2*np.pi, nz)
uvel = np.real(u['g'].reshape(nz, 1)* np.exp(1j*ly.reshape(1,nz)))
vvel = np.real(v['g'].reshape(nz, 1)* np.exp(1j*ly.reshape(1,nz)))
wvel = np.real(w['g'].reshape(nz, 1)* np.exp(1j*ly.reshape(1,nz)))
maxu = np.max(uvel)

uvel = uvel/maxu
vvel = vvel/maxu
wvel = wvel/maxu
buoy = np.real(b['g'].reshape(nz, 1)* np.exp(1j*ly.reshape(1,nz)))
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
vs = .85
#vs = 0.12
im = ax2.contourf(ly, z, vvel, np.linspace(-vs, vs, nc),vmin=-vs, vmax=vs, cmap=cm)
cb = plt.colorbar(im, ax=ax2)
cb.set_ticks([-vs, 0, vs])
ax2.contour(ly, z, vvel, np.linspace(-vs,vs, nc), colors='0.8', linewidths=lw)

ax2.set_title('Along-slope velocity', fontsize=fs)
ax2.grid(linestyle='--', alpha = 0.5)

# WVEL
ws = 0.05
#ws = 0.01
im = ax3.contourf(ly, z, wvel, np.linspace(-ws, ws, nc),vmin=-ws, vmax=ws, cmap=cm)
cb = plt.colorbar(im, ax=ax3)
cb.set_ticks([-ws, 0, ws])
ax3.contour(ly, z, wvel, np.linspace(-ws, ws, nc), colors='0.8', linewidths=lw)

ax3.set_title('Slope-normal velocity', fontsize=fs)
ax3.grid(linestyle='--', alpha = 0.5)

# BUOY
im = ax4.contourf(ly, z, buoy, np.linspace(-1, 1, nc),vmin=-1, vmax=1, cmap=cm)
cb = plt.colorbar(im, ax=ax4)
ax4.grid(linestyle='--', alpha = 0.5)
ax4.contour(ly, z, buoy, np.linspace(-1, 1, nc), colors='0.8', linewidths=lw)

#np.linspace(-0.15, 0.15, nc)
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
ax5.set_ylim((0, 150))
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
#plt.savefig('/home/jacob/Dropbox/Slope BI/Slope BI Manuscript/EkmanPerturbationsCombo.pdf', format='pdf', bbox_inches='tight')
#fig.subplots_adjust(wspace=10, vspace=0.1)
plt.show()

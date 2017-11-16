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
plt.rc('text', usetex=True)
import dedalus.public as de
from mpi4py import MPI
CW = MPI.COMM_WORLD
from pylab import *
import scipy.integrate as integrate

import logging
logger = logging.getLogger(__name__)


# Global parameters
directoryname = "/home/jacob/dedalus/SlopeAngleRi1/"

# Variable Parameters
ln = 2.1# 0.25
delta = -0.5

# Physical parameters
f = 1e-4
tht = 0
Pr = 1
H = 100
H = 1000
BLH = 250
Ri = 100
RiBL = 1
#Bzmag = 2.5e-5
#Shmag = np.sqrt(Bzmag/Ri)
Shmag = .1/H
Bzmag = Ri*Shmag**2
BzmagBL = RiBL*Shmag**2

l = ln*f/(H*np.sqrt(Bzmag))
l = 0.0040392358531509045
# Grid Parameters
nz = 128#256

# Create bases and domain
# Use COMM_SELF so keep calculations independent between processes
z_basis = de.Chebyshev('z', nz, interval=(0,H))
domain = de.Domain([z_basis], grid_dtype=np.complex128, comm=MPI.COMM_SELF)

z = domain.grid(0)

# Define Stability Analysis Parameters

kap = domain.new_field(name='kap')
kap['g'] = 1e-5*np.ones(z.shape)
U = domain.new_field(name='U')
U['g'] = 0*z
Uz = domain.new_field(name='Uz')
Uz['g'] = 0*z
V = domain.new_field(name='V')
Vz = domain.new_field(name='Vz')
Bz = domain.new_field(name='Bz')
B = domain.new_field(name='B')
V['g'] = Shmag*(z)
Vz['g'] = Shmag*(z-z+1) #Note this assumes no horizotal variation (ie. won't work for the non-uniform case)
Bt = np.zeros([nz])
Bz['g'] = np.array(Bzmag*np.ones([nz]))
zind = np.floor( next((x[0] for x in enumerate(z) if x[1]>BLH)))
#Bz['g'][0:zind] = BzmagBL

tpoint = np.floor( next((x[0] for x in enumerate(z) if x[1]>BLH)))
Bstr  = -0.5*(np.tanh((-z + z[tpoint])/40)+1)
Bz['g'] = Bz['g']*10**(2*Bstr)
    
    
Bt[1:nz] = integrate.cumtrapz(Bz['g'], z)
B['g'] = Bt
# 2D Boussinesq hydrodynamics, with no-slip boundary conditions
# Use substitutions for x and t derivatives
#    bz = 1/(2*Ri*np.sin(tht)**2)*(f**2*np.cos(tht)+np.sqrt(f**4*np.cos(tht)**2 + 4*Bzmag*f**2*Ri*np.sin(tht)**2))
#    Shmag = -bz/(f*np.sin(tht))

tht = delta/(Bzmag/(f*Shmag))
tht = 5e-3

#problem = de.EVP(domain, variables=['u', 'v', 'w', 'b', 'p', 'uz', 'vz', 'wz',
#        'bz'], eigenvalue='omg', tolerance = 1e-10)
#problem.parameters['tht'] = tht
#problem.parameters['U'] = U
#problem.parameters['V'] = V
#problem.parameters['B'] = B
#problem.parameters['Uz'] = Uz
#problem.parameters['Vz'] = Vz
#problem.parameters['NS'] = Bz
#problem.parameters['f'] = f
#problem.parameters['tht'] = tht
#problem.parameters['kap'] = kap
#problem.parameters['Pr'] = Pr
#problem.parameters['k'] = 0. # will be set in loop
#problem.parameters['l'] = l # will be set in loop
#problem.substitutions['dx(A)'] = "1j*k*A"
#problem.substitutions['dy(A)'] = "1j*l*A"
#problem.substitutions['dt(A)'] = "-1j*omg*A"
#problem.add_equation(('dt(u) + U*dx(u) + V*dy(u) + w*Uz - f*v*cos(tht) + dx(p)'
#        '- b*sin(tht) - Pr*(kap*dx(dx(u)) + kap*dy(dy(u)) + dz(kap)*uz'
#        '+ kap*dz(uz)) = 0'))
#problem.add_equation(('dt(v) + U*dx(v) + V*dy(v) + w*Vz*cos(tht) + f*u*cos(tht)'
#        '- f*w*sin(tht) + dy(p) - Pr*(kap*dx(dx(v)) + kap*dy(dy(v))'
#        '+ dz(kap)*vz + kap*dz(vz)) = 0'))
#problem.add_equation(('(dt(w) + U*dx(w) + V*dy(w)) + f*v*sin(tht) + dz(p)'
#        '- b*cos(tht) - Pr*(kap*dx(dx(w)) + kap*dy(dy(w)) + dz(kap)*wz'
#        '+ kap*dz(wz)) = 0'))
##    problem.add_equation(('dt(b) + U*dx(b) + V*dy(b) + u*(NS*sin(tht))'
##                '+ w*(NS*cos(tht)) - kap*dx(dx(b)) - kap*dy(dy(b)) - dz(kap)*bz'
##                '- kap*dz(bz) = 0'))
#problem.add_equation(('dt(b) + U*dx(b) + V*dy(b) + u*(NS*sin(tht)+f*Vz*cos(tht))'
#        '+ w*(NS*cos(tht)-f*Vz*sin(tht)) - kap*dx(dx(b)) - kap*dy(dy(b)) - dz(kap)*bz'
#        '- kap*dz(bz) = 0'))
##problem.add_equation(('dt(b) + U*dx(b) + V*dy(b) + u*Vz*f'
##        '+ w*(Bz) - kap*dx(dx(b)) - kap*dy(dy(b)) - dz(kap)*bz'
##        '- kap*dz(bz) = 0'))
#problem.add_equation('dx(u) + dy(v) + wz = 0')
#problem.add_equation('uz - dz(u) = 0')
#problem.add_equation('vz - dz(v) = 0')
#problem.add_equation('wz - dz(w) = 0')
#problem.add_equation('bz - dz(b) = 0')
#problem.add_bc('left(u) = 0')
#problem.add_bc('left(v) = 0')
#problem.add_bc('left(w) = 0')
#problem.add_bc('left(bz) = 0')
#problem.add_bc('right(uz) = 0')
#problem.add_bc('right(vz) = 0')
#problem.add_bc('right(w) = 0')
##    problem.add_bc('right(w) = 1/10*(dt(right(p)) + right(u)*dx(right(p))+right(v)*dy(right(p)))')
#problem.add_bc('right(bz) = 0')

problem = de.EVP(domain, variables=['u', 'v', 'w', 'b', 'p'], eigenvalue='omg', tolerance = 1e-10)
problem.parameters['tht'] = tht
problem.parameters['V'] = V
problem.parameters['Uz'] = Uz
problem.parameters['Vz'] = Vz
problem.parameters['NS'] = Bz
problem.parameters['f'] = f
problem.parameters['tht'] = tht
problem.parameters['kap'] = kap
problem.parameters['Pr'] = Pr
problem.parameters['k'] = 0. # will be set in loop
problem.parameters['l'] = l # will be set in loop
problem.substitutions['dx(A)'] = "1j*k*A"
problem.substitutions['dy(A)'] = "1j*l*A"
problem.substitutions['dt(A)'] = "-1j*omg*A"
problem.add_equation(('dt(u) + V*dy(u) - f*v*cos(tht) + dx(p)- b*sin(tht) = 0'))
problem.add_equation(('dt(v) + V*dy(v) + w*Vz/cos(tht) + f*u*cos(tht)- f*w*sin(tht) + dy(p) = 0'))
problem.add_equation(('(dt(w) + V*dy(w)) + f*v*sin(tht) + dz(p)- b*cos(tht) = 0'))
problem.add_equation(('dt(b) +  V*dy(b) + u*(NS*sin(tht)+f*Vz*cos(tht))'
        '+ w*(NS*cos(tht)-f*Vz*sin(tht)) = 0'))
problem.add_equation('dx(u) + dy(v) + dz(w) = 0')

problem.add_bc('left(w) = 0')
problem.add_bc('right(w) = 0')


solver = problem.build_solver()
    
#%%
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
v = solver.state['v']
w = solver.state['w']
b = solver.state['b']
#%%
# shear production
VSP = -2*np.real(np.conj(w['g'])*v['g']*Vz['g'] + np.conj(v['g'])*u['g']*Vz['g']*np.sin(tht))
Vx = Vz['g']*np.sin(tht)
LSP = -2*np.real(-np.conj(w['g'])*v['g']*Vx*np.sin(tht) + np.conj(v['g'])*u['g']*Vx)

# buoyancy production
BP = 2*np.real((u['g']*np.sin(tht)+w['g']*np.cos(tht))*np.conj(b['g']))

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
fs = 16
plt.figure(figsize=(5, 5))
plt.plot(BP/np.max(BP), z)
plt.plot((VSP+LSP)/np.max(BP), z)
#plt.plot(LSP/np.max(BP), z)

plt.xlabel('Kinetic Energy Tendency', fontsize=fs)
plt.ylabel('slope-normal coordinate [m]', fontsize=fs)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
plt.legend(['Buoyancy Production', 'Shear Production'], frameon=False, fontsize=fs, loc=1)
plt.tight_layout()
plt.ylim((0, 1000))
plt.grid(linestyle='--', alpha = 0.5)

#plt.xlim((-0.1, 1))
#plt.savefig('/home/jacob/Dropbox/Slope BI/Slope BI Manuscript/IdealizedEnergetics.eps', format='eps', dpi=1000, bbox_inches='tight')

#%%
# most unstable mode
nc  = 40
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

fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 10))
# UVEL
im = ax[0,0].contourf(ly, z, uvel, np.linspace(-1, 1, nc),vmin=-1, vmax=1, cmap='RdBu_r')
cb = plt.colorbar(im, ax=ax[0,0])
cb.set_ticks([-1, 0, 1])
ax[0,0].set_title('Across-slope velocity', fontsize=fs)
ax[0,0].grid(linestyle='--', alpha = 0.5)

# VVEL
im = ax[0,1].contourf(ly, z, vvel, np.linspace(-0.6, 0.6, nc),vmin=-0.6, vmax=0.6, cmap='RdBu_r')
cb = plt.colorbar(im, ax=ax[0,1])
cb.set_ticks([-0.6, 0, 0.6])
ax[0,1].set_title('Along-slope velocity', fontsize=fs)
ax[0,1].grid(linestyle='--', alpha = 0.5)

# WVEL
im = ax[1,0].contourf(ly, z, wvel, np.linspace(-0.15, 0.15, nc),vmin=-0.15, vmax=0.15, cmap='RdBu_r')
cb = plt.colorbar(im, ax=ax[1,0])
cb.set_ticks([-0.15, 0, 0.15])
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
#plt.savefig('/home/jacob/Dropbox/Slope BI/Slope BI Manuscript/IdealizedPerturbations.pdf', format='pdf')

plt.show()

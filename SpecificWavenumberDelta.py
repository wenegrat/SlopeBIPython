"""
Dedalus script for plotting the structure of a specific wavenumber/delta
To run using 4 processes, for instance, you could use:
    $ mpiexec -n 4 python3 SpecificWavenumberDelta.py

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
import scipy.integrate as integrate

import logging
logger = logging.getLogger(__name__)


# Global parameters
# Physical parameters
f = 1e-4
Pr = 1
H = 1000
BLH = 250
Ri = 100
RiBL = 1

Shmag = .1/H
Bzmag = Ri*Shmag**2
BzmagBL = RiBL*Shmag**2

l = 0.0040392358531509045 # Pick specific wavenumber
#l = 0.00075*2*np.pi
#l = 0.00022539339047347913 #deep mode
#l = 0.009
#l = 1e-6

tht = 5e-3

# Grid Parameters
nz = 256

# Create bases and domain
# Use COMM_SELF so keep calculations independent between processes
z_basis = de.Chebyshev('z', nz, interval=(0,H))
domain = de.Domain([z_basis], grid_dtype=np.complex128, comm=MPI.COMM_SELF)

z = domain.grid(0)

# Define Stability Analysis Parameters


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
Bz['g'][0:zind] = BzmagBL

tpoint = np.floor( next((x[0] for x in enumerate(z) if x[1]>BLH)))
Bstr  = -0.5*(np.tanh((-z + z[tpoint])/40)+1)

Bz['g'] = Bz['g']*10**(2*Bstr)    
Bt[1:nz] = integrate.cumtrapz(Bz['g'], z)
B['g'] = Bt


problem = de.EVP(domain, variables=['u', 'v', 'w', 'b', 'p'], eigenvalue='omg', tolerance = 1e-10)
problem.parameters['tht'] = tht
problem.parameters['V'] = V
problem.parameters['Uz'] = Uz
problem.parameters['Vz'] = Vz
problem.parameters['NS'] = Bz
problem.parameters['f'] = f
problem.parameters['tht'] = tht
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
problem.add_bc('right(w) = -right(u)*tan(tht)')


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
cm = 'RdBu_r'
# most unstable mode
fs = 18

plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': fs})
plt.rcParams['contour.negative_linestyle'] = 'solid'

nc  = 16
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
vs = 0.6
#vs = 0.12
im = ax2.contourf(ly, z, vvel, np.linspace(-vs, vs, nc),vmin=-vs, vmax=vs, cmap=cm)
cb = plt.colorbar(im, ax=ax2)
cb.set_ticks([-vs, 0, vs])
ax2.contour(ly, z, vvel, np.linspace(-vs,vs, nc), colors='0.8', linewidths=lw)

ax2.set_title('Along-slope velocity', fontsize=fs)
ax2.grid(linestyle='--', alpha = 0.5)

# WVEL
ws = 0.15
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
ax5.plot((VSP+LSP)/np.max(BP), z)
#plt.plot(LSP/np.max(BP), z)

ax5.set_xlabel('Kinetic energy tendency', fontsize=fs)
ax5.set_ylabel('Slope-normal coordinate [m]', fontsize=fs)
ax5.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
leg = ax5.legend(['Buoyancy Production', 'Shear Production'], frameon=True, fontsize=12, loc=9)
leg.get_frame().set_alpha(.9)
#plt.tight_layout()
ax5.set_ylim((0, 1000))
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
#plt.savefig('/home/jacob/Dropbox/Slope BI/Slope BI Manuscript/IdealizedPerturbationsCombo.pdf', format='pdf', bbox_inches='tight')
#fig.subplots_adjust(wspace=10, vspace=0.1)
plt.show()

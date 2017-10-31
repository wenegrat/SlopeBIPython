import numpy as np
import matplotlib.pyplot as plt
from pylab import *

from dedalus import public as de

import logging
logger = logging.getLogger(__name__)

# parameters
N = 1e-3 # buoyancy frequency
f = 2.5e-5 # Coriolis parameter
tht = 2e-3 # slope angle
kap_0 = 1e-5 # background diffusivity
kap_1 = 5e-3 # bottom enhancement of diffusivity
h = 200. # decay scale of mixing
Pr = 1 # Prandtl number
H = 2500. # domain height

# along-slope wavenumbers
ll = np.logspace(-5, -2, 32)

# number of grid points
nz = 64

# file name that results are saved in
name = 'test'

# build domain
z_basis = de.Chebyshev('z', nz, interval=(0, H))
domain = de.Domain([z_basis], np.complex128)

# non-constant coefficients
kap = domain.new_field(name='kap')
z = domain.grid(0)
kap['g'] = kap_0 + kap_1*np.exp(-z/h)

# STEADY STATE

# setup problem
problem = de.LBVP(domain, variables=['U', 'V', 'B', 'Uz', 'Vz', 'Bz'])
problem.parameters['N'] = N
problem.parameters['f'] = f
problem.parameters['tht'] = tht
problem.parameters['kap'] = kap
problem.parameters['Pr'] = Pr
problem.add_equation(('-f*V*cos(tht) - B*sin(tht) - Pr*(dz(kap)*Uz'
        '+ kap*dz(Uz)) = 0'))
problem.add_equation('f*U*cos(tht) - Pr*(dz(kap)*Vz + kap*dz(Vz)) = 0')
problem.add_equation(('U*N**2*sin(tht) - dz(kap)*Bz - kap*dz(Bz)'
        '= dz(kap)*N**2*cos(tht)'))
problem.add_equation('Uz - dz(U) = 0')
problem.add_equation('Vz - dz(V) = 0')
problem.add_equation('Bz - dz(B) = 0')
problem.add_bc('left(U) = 0')
problem.add_bc('left(V) = 0')
problem.add_bc('left(Bz) = -N**2*cos(tht)')
problem.add_bc('right(Uz) = 0')
problem.add_bc('right(Vz) = 0')
problem.add_bc('right(Bz) = 0')

# build solver and solve
solver = problem.build_solver()
solver.solve()

# collect solution
U = solver.state['U']
V = solver.state['V']
B = solver.state['B']
Uz = solver.state['Uz']
Vz = solver.state['Vz']
Bz = solver.state['Bz']

#%% ADDING MY OWN CHECK OF RI
N2hat = Bz['g'] + N**2*cos(tht)
M2hat = N**2*sin(tht)
N2 = N2hat*cos(tht) + M2hat*sin(tht)
M2 = M2hat*cos(tht) - N2hat*sin(tht)
Ri = N2*f**2/M2**2
dt = np.sqrt(2*kap_1/f)

plt.figure()
plt.plot(Ri[0:10], z[0:10])
plt.axhline(y=dt, color='r', linestyle='-')
#%%
# LINEAR STABILITY

problem = de.EVP(domain, variables=['u', 'v', 'w', 'b', 'p', 'wz'], eigenvalue='omg')
problem.parameters['U'] = U
problem.parameters['V'] = V
problem.parameters['B'] = B
problem.parameters['Uz'] = Uz
problem.parameters['Vz'] = Vz
problem.parameters['Bz'] = Bz
problem.parameters['N'] = N
problem.parameters['f'] = f
problem.parameters['tht'] = tht
problem.parameters['kap'] = kap
problem.parameters['Pr'] = Pr
problem.parameters['k'] = 0. # will be set in loop
problem.parameters['l'] = 0. # will be set in loop
problem.substitutions['dx(A)'] = "1j*k*A"
problem.substitutions['dy(A)'] = "1j*l*A"
problem.substitutions['dt(A)'] = "-1j*omg*A"
problem.add_equation(('dt(u) + U*dx(u) + V*dy(u) + w*Uz - f*v*cos(tht) + dx(p)'
        '- b*sin(tht)  = 0'))
problem.add_equation(('dt(v) + U*dx(v) + V*dy(v) + w*Vz + f*u*cos(tht)'
        '- f*w*sin(tht) + dy(p) = 0'))
problem.add_equation(('dt(w) + U*dx(w) + V*dy(w) + f*v*sin(tht) + dz(p)'
        '- b*cos(tht)  = 0'))
problem.add_equation(('dt(b) + U*dx(b) + V*dy(b) + u*N**2*sin(tht)'
        '+ w*(N**2*cos(tht) + Bz) = 0'))
problem.add_equation('dx(u) + dy(v) + wz = 0')
#problem.add_equation('uz - dz(u) = 0')
#problem.add_equation('vz - dz(v) = 0')
problem.add_equation('wz - dz(w) = 0')
#problem.add_equation('bz - dz(b) = 0')
#problem.add_bc('left(u) = 0')
#problem.add_bc('left(v) = 0')
problem.add_bc('left(w) = 0')
#problem.add_bc('left(bz) = 0')
#problem.add_bc('right(uz) = 0')
#problem.add_bc('right(vz) = 0')
problem.add_bc('right(w) = 0')
#problem.add_bc('right(bz) = 0')

# set up solver
solver = problem.build_solver()
#%%
def sorted_eigen(k, l):

    """
    Solves eigenvalue problem and returns sorted eigenvalues and associated
    eigenvectors.
    """

    # set wavenumbers
    problem.namespace['k'].value = k
    problem.namespace['l'].value = l

    # solve problem
    solver.solve_dense(solver.pencils[0], rebuild_coeffs=True)

    # sort eigenvalues
    omg = solver.eigenvalues
    omg[np.isnan(omg)] = 0.
    omg[np.isinf(omg)] = 0.
    idx = np.argsort(omg.imag)

    return idx

def max_growth_rate(k, l):

    """Finds maximum growth rate for given wavenumbers k, l."""

    print(k, l)

    # solve eigenvalue problem and sort
    idx = sorted_eigen(k, l)

    return solver.eigenvalues[idx[-1]].imag

# get max growth rates
gr = np.array([max_growth_rate(0, l) for l in ll])

# get full eigenvectors and eigenvalues for l with largest growth
idx = sorted_eigen(0., ll[np.argmax(gr)])
solver.set_state(idx[-1])

# collect eigenvector
u = solver.state['u']
v = solver.state['v']
w = solver.state['w']
b = solver.state['b']

# shear production
SP = -2*np.real(np.conj(w['g'])*(u['g']*Uz['g']+v['g']*Vz['g']))

# buoyancy production
BP = 2*np.real((u['g']*np.sin(tht)+w['g']*np.cos(tht))*np.conj(b['g']))

# SAVE TO FILE

np.savez(name + '.npz', nz=nz, N=N, tht=tht, z=z, kap=kap['g'], Pr=Pr, U=U['g'],
        V=V['g'], B=B['g'], u=u['g'], v=v['g'], w=w['g'], b=b['g'], ll=ll,
        gr=gr, SP=SP, BP=BP)

# PLOTTING

# mean state

fig, ax = plt.subplots(1, 3, sharey=True)

ax[0].semilogx(kap['g'], z)
ax[0].set_xlabel('mixing coefficient [m$^2$/s]', va='baseline')
ax[0].set_ylabel('slope-normal coordinate [m]')
ax[0].get_xaxis().set_label_coords(.5, -.12)

ax[1].plot(U['g'].real, z)
ax[1].plot(V['g'].real, z)
ax[1].set_xlabel('mean flow [m/s]', va='baseline')
ax[1].get_xaxis().set_label_coords(.5, -.12)

ax[2].plot(N**2*np.cos(tht)*z + B['g'].real, z)
ax[2].set_xlabel('mean buoyancy [m/s$^2$]', va='baseline')
ax[2].get_xaxis().set_label_coords(.5, -.12)

fig.savefig('fig/mean_state.pdf')

# energetics
#%%
plt.figure(figsize=(4.8, 4.8))
plt.semilogx(ll, gr)
plt.xlabel('along-track wavenumber [m$^{-1}$]')
plt.ylabel('growth rate')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.tight_layout()
#plt.savefig('fig/growth_rate.pdf')
#%%
plt.figure(figsize=(4.8, 4.8))
plt.plot(SP, z)
plt.plot(BP, z)
plt.xlabel('kinetic energy tendency [m$^2$/s$^3$]')
plt.ylabel('slope-normal coordinate [m]')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
plt.legend(['shear production', 'buoyancy production'], frameon=False)
plt.tight_layout()
#plt.savefig('fig/energetics.pdf')

# most unstable mode

ly = np.linspace(0, 2*np.pi, nz)

fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(6.4, 6.4))
im = ax[0,0].pcolormesh(ly, z, np.real(u['g'].reshape(nz, 1)
        * np.exp(1j*ly.reshape(1,nz))), rasterized=True, cmap='RdBu_r')
plt.colorbar(im, ax=ax[0,0])
ax[0,0].set_title('across-slope velocity')
im = ax[0,1].pcolormesh(ly, z, np.real(v['g'].reshape(nz, 1)
        * np.exp(1j*ly.reshape(1,nz))), rasterized=True, cmap='RdBu_r')
plt.colorbar(im, ax=ax[0,1])
ax[0,1].set_title('along-slope velocity')
im = ax[1,0].pcolormesh(ly, z, np.real(w['g'].reshape(nz, 1)
        * np.exp(1j*ly.reshape(1,nz))), rasterized=True, cmap='RdBu_r')
plt.colorbar(im, ax=ax[1,0])
ax[1,0].set_title('slope-normal velocity')
im = ax[1,1].pcolormesh(ly, z, np.real(b['g'].reshape(nz, 1)
        * np.exp(1j*ly.reshape(1,nz))), rasterized=True, cmap='RdBu_r')
plt.colorbar(im, ax=ax[1,1])
ax[1,1].set_title('buoyancy')
ax[0,0].set_xticks([0, np.pi, 2*np.pi])
ax[1,0].set_xlabel('phase')
ax[1,1].set_xlabel('phase')
ax[0,0].set_ylabel('slope-normal coordinate [m]')
ax[1,0].set_ylabel('slope-normal coordinate [m]')
#plt.savefig('fig/modes.pdf', dpi=300)

plt.show()

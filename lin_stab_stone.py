import numpy as np
import matplotlib.pyplot as plt

from dedalus import public as de

import logging
logger = logging.getLogger(__name__)

# parameters
N = 1e-3 # buoyancy frequency
f = 1e-4 # Coriolis parameter
tht = -5e-2 # slope angle
H = 100. # domain height
Lmd = 1e-3 # shear

# slope parameter
print(np.tan(tht)*N**2/(f*Lmd))

# along-slope wavenumbers
ll = np.linspace(.01*f/(N*H), 4*f/(N*H), 16)

# number of grid points
nz = 128

# file name that results are saved in
name = 'test'

# build domain
z_basis = de.Chebyshev('z', nz, interval=(0, H))
domain = de.Domain([z_basis], np.complex128)

# LINEAR STABILITY

problem = de.EVP(domain, variables=['u', 'v', 'w', 'b', 'p'], eigenvalue='omg')
problem.parameters['N'] = N
problem.parameters['f'] = f
problem.parameters['tht'] = tht
problem.parameters['Lmd'] = Lmd
problem.parameters['k'] = 0. # will be set in loop
problem.parameters['l'] = 0. # will be set in loop
problem.substitutions['dx(A)'] = "1j*k*A"
problem.substitutions['dy(A)'] = "1j*l*A"
problem.substitutions['dt(A)'] = "-1j*omg*A"
problem.add_equation('dt(u) + Lmd*z/cos(tht)*dy(u) - f*v*cos(tht) + dx(p) - b*sin(tht) = 0')
problem.add_equation('dt(v) + Lmd*z/cos(tht)*dy(v) + w*Lmd/cos(tht) + f*u*cos(tht) - f*w*sin(tht) + dy(p) = 0')
problem.add_equation('dt(w) + Lmd*z/cos(tht)*dy(w) + f*v*sin(tht) + dz(p) - b*cos(tht) = 0')
problem.add_equation('dt(b) + Lmd*z/cos(tht)*dy(b) + u*(N**2*sin(tht) + f*Lmd*cos(tht)) + w*(N**2*cos(tht) - f*Lmd*sin(tht)) = 0')
problem.add_equation('dx(u) + dy(v) + dz(w) = 0')
problem.add_bc('left(w) = 0')
problem.add_bc('right(w) = 0')

# set up solver
solver = problem.build_solver()

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

z = domain.grid(0)

# PLOTTING

# mean state

plt.figure(figsize=(4.8, 4.8))
plt.plot(ll*abs(Lmd*H/f), gr/abs(f))
plt.xlabel('along-slope wavenumber')
plt.ylabel('growth rate')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.tight_layout()
plt.savefig('fig/growth_rate.pdf')

# most unstable mode

ly = np.linspace(0, 2*np.pi, nz)

fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(6.4, 6.4))
im = ax[0,0].pcolormesh(ly, z/H, np.real(u['g'].reshape(nz, 1) * np.exp(1j*ly.reshape(1,nz))), rasterized=True, cmap='RdBu_r')
plt.colorbar(im, ax=ax[0,0])
ax[0,0].set_title('across-slope velocity')
im = ax[0,1].pcolormesh(ly, z/H, np.real(v['g'].reshape(nz, 1) * np.exp(1j*ly.reshape(1,nz))), rasterized=True, cmap='RdBu_r')
plt.colorbar(im, ax=ax[0,1])
ax[0,1].set_title('along-slope velocity')
im = ax[1,0].pcolormesh(ly, z/H, np.real(w['g'].reshape(nz, 1) * np.exp(1j*ly.reshape(1,nz))), rasterized=True, cmap='RdBu_r')
plt.colorbar(im, ax=ax[1,0])
ax[1,0].set_title('slope-normal velocity')
im = ax[1,1].pcolormesh(ly, z/H, np.real(b['g'].reshape(nz, 1) * np.exp(1j*ly.reshape(1,nz))), rasterized=True, cmap='RdBu_r')
plt.colorbar(im, ax=ax[1,1])
ax[1,1].set_title('buoyancy')
ax[0,0].set_xticks([0, np.pi, 2*np.pi])
ax[1,0].set_xlabel('phase')
ax[1,1].set_xlabel('phase')
ax[0,0].set_ylabel('slope-normal coordinate')
ax[1,0].set_ylabel('slope-normal coordinate')
plt.savefig('fig/modes.pdf', dpi=300)

plt.show()

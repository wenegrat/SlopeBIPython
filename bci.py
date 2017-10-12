# Baroclinic instability test case

import numpy as np
import matplotlib.pyplot as plt
import h5py
from dedalus import public as de
from dedalus.extras import flow_tools
import time
import logging

nx = 32
ny = 32
nz = 32

eps = .05
lmd = .2
dlt = 1.
sig = 1.
gam = 5e-3

dt = 1e-2

name = 'run_{:4.2f}_{:3.1f}_{:3.1f}_{:6.1e}_{:6.1e}'.format(eps, lmd, dlt, gam, dt)
print(name)

root = logging.root
for h in root.handlers:
    h.setLevel("INFO")
    
logger = logging.getLogger(__name__)

x_basis = de.Fourier('x', nx, interval=(0, 1), dealias=3/2)
y_basis = de.Fourier('y', ny, interval=(0, 1), dealias=3/2)
z_basis = de.Chebyshev('z', nz, interval=(0, 1), dealias=3/2)
domain = de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.float64)

problem = de.IVP(domain, variables=['u', 'v', 'w', 'b', 'p', 'uz', 'vz', 'wz', 'bz'])

problem.parameters['eps'] = eps
problem.parameters['lmd'] = lmd
problem.parameters['dlt'] = dlt
problem.parameters['sig'] = sig
problem.parameters['gam'] = gam

problem.substitutions['D(A,Az)'] = 'gam*(dlt**2*(dx(dx(A)) + dy(dy(A))) + dz(Az))'
problem.substitutions['N(A,Az)'] = 'u*dx(A) + v*dy(A) + w*Az'

problem.add_equation('       eps**2*(dt(u) + lmd*z*dx(u) + lmd*w - sig*D(u,uz)) - v + dx(p)     =        -eps**2*N(u,uz)')
problem.add_equation('       eps**2*(dt(v) + lmd*z*dx(v)         - sig*D(v,vz)) + u + dy(p)     =        -eps**2*N(v,vz)')
problem.add_equation('dlt**2*eps**2*(dt(w) + lmd*z*dx(w)         - sig*D(w,wz))     + dz(p) - b = -dlt**2*eps**2*N(w,wz)')
problem.add_equation('dx(u) + dy(v) + wz = 0')
problem.add_equation('dt(b) + lmd*z*dx(b) - lmd*v + w - D(b,bz) = -N(b,bz)')

problem.add_equation('dz(u) - uz = 0')
problem.add_equation('dz(v) - vz = 0')
problem.add_equation('dz(w) - wz = 0')
problem.add_equation('dz(b) - bz = 0')

problem.add_bc("left(uz) = 0")
problem.add_bc("right(uz) = 0")
problem.add_bc("left(vz) = 0")
problem.add_bc("right(vz) = 0")
problem.add_bc("left(w) = 0")
problem.add_bc("right(w) = 0", condition="(ny != 0)")
problem.add_bc("integ(p,'z') = 0", condition="(ny == 0)")
problem.add_bc("left(bz) = 0")
problem.add_bc("right(bz) = 0")

# time stepping
ts = de.timesteppers.RK443

solver = problem.build_solver(ts)

x = domain.grid(0)
y = domain.grid(1)
z = domain.grid(2)
u = solver.state['u']
v = solver.state['v']
w = solver.state['w']
b = solver.state['b']
p = solver.state['p']

# initial conditions
u['g'] = 0
v['g'] = 0
w['g'] = 0
b['g'] = 1e-4*np.sqrt(nx*ny*nz)*np.random.randn(nx,ny,nz)

# flow control
solver.stop_sim_time = 1e9
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf

# plot setup
x = domain.grid(0, scales=domain.dealias)
y = domain.grid(1, scales=domain.dealias)
z = domain.grid(2, scales=domain.dealias)

logger.info('Starting loop')
start_time = time.time()

while solver.ok:

    # time step
    solver.step(dt)

    if solver.iteration % 10 == 0:
        logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))

    if solver.iteration % 10 == 0:
        # Make plot of u at surface
        plt.figure()
        plt.pcolormesh(x[:,0,0], y[0,:,0], u['g'][:,:,-1].T, Rasterized=True)
        plt.colorbar()
        plt.savefig(name + '/u_surf_{:010d}.png'.format(solver.iteration), dpi=300)
        plt.close()
        # Make plot of v at surface
        plt.figure()
        plt.pcolormesh(x[:,0,0], y[0,:,0], v['g'][:,:,-1].T, Rasterized=True)
        plt.colorbar()
        plt.savefig(name + '/v_surf_{:010d}.png'.format(solver.iteration), dpi=300)
        plt.close()
        # Make plot of b at surface
        plt.figure()
        plt.pcolormesh(x[:,0,0], y[0,:,0], (-lmd*(y-.5) + b['g'])[:,:,-1].T, rasterized=True)
        plt.colorbar()
        plt.clim(-lmd, lmd)
        plt.savefig(name + '/b_surf_{:010d}.png'.format(solver.iteration), dpi=300)
        plt.close()

end_time = time.time()

# Print statistics
logger.info('Run time: %f' %(end_time-start_time))
logger.info('Iterations: %i' %solver.iteration)

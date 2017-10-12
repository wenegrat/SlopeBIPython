#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 10:24:38 2017

@author: jacob
"""

"""
Dedalus script for 3D Rayleigh-Benard convection.

This script uses parity-bases in the x and y directions to mimick no-slip,
insulating sidewalls.  The equations are scaled in units of the thermal
diffusion time (Pe = 1).

This script should be ran in parallel, and would be most efficient using a
2D process mesh.  It uses the built-in analysis framework to save 2D data slices
in HDF5 files.  The `merge.py` script in this folder can be used to merge
distributed analysis sets from parallel runs, and the `plot_2d_series.py` script
can be used to plot the slices.

To run, merge, and plot using 4 processes, for instance, you could use:
    $ mpiexec -n 4 python3 rayleigh_benard.py
    $ mpiexec -n 4 python3 merge.py snapshots
    $ mpiexec -n 4 python3 plot_2d_series.py snapshots/*.h5

The simulation should take roughly 400 process-minutes to run, but will
automatically stop after an hour.

"""

import numpy as np
from mpi4py import MPI
import time
from pylab import *
from dedalus import public as de
from dedalus.extras import flow_tools

import logging
logger = logging.getLogger(__name__)


# Parameters
Lx, Ly, Lz = (5e3,5e3, 100.)
f = 1e-4 # Coriolis parameter
N2 = (12*f)**2
M2 = (6*f)**2
tht = 2e-3 # slope angle
kapp = 1e-5
nu = 1e-5

# Create bases and domain
start_init_time = time.time()
x_basis = de.Fourier('x', 64, interval=(0, Lx), dealias=3/2)
y_basis = de.Fourier('y', 64, interval=(0, Ly), dealias=3/2)
z_basis = de.Chebyshev('z', 32, interval=(0, Lz), dealias=3/2)
domain = de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.float64)

# non-constant coefficients
Vg = domain.new_field(name='Vg')
z = domain.grid(2)
Vg['g'] = M2/f*(z)

# 2D Boussinesq hydrodynamics
problem = de.IVP(domain, variables=['p','b','u','v','w','bz','uz','vz', 'wz'], time='t')
problem.parameters['f'] = f
problem.parameters['kap'] = kapp
problem.parameters['nu'] = nu
problem.parameters['Ms'] = M2
problem.parameters['Vg'] = Vg

#problem.add_equation("p = 0" , condition="(nx==0) and (ny==0)")
problem.add_equation("dx(u) + dy(v) + wz = 0")
problem.add_equation("dt(b) - kap*(dx(dx(b)) + dy(dy(b)) + dz(bz))  = - u*(dx(b)+Ms) - v*dy(b) - w*bz - Vg*dy(b) ")
problem.add_equation("dt(u) - f*v + dx(p)  - nu*(dx(dx(u)) + dy(dy(v)) + dz(uz))   = - u*dx(u) - v*dy(u) - w*uz - Vg*dy(u)")
problem.add_equation("dt(v) + f*u + dy(p)  - nu*(dx(dx(v)) + dy(dy(v)) + dz(vz))  = - u*dx(v) - v*dy(v) - w*(vz+ Ms/f)  - Vg*dy(v)")
problem.add_equation("dt(w) + dz(p) - b -nu*(dx(dx(w)) + dy(dy(w))+ dz(wz)) = -u*dx(w) - v*dy(w)- w*wz - Vg*dy(w)")
problem.add_equation("bz - dz(b) = 0")
problem.add_equation("uz - dz(u) = 0")
problem.add_equation("vz - dz(v) = 0")
problem.add_equation("wz - dz(w) = 0")

problem.add_bc("left(bz)= 0")
problem.add_bc("left(u) = 0")
problem.add_bc("left(v) = 0")
problem.add_bc("left(w) = 0")
problem.add_bc("right(bz) = 0")
problem.add_bc("right(uz) = 0")
problem.add_bc("right(vz) = -Ms/f")
problem.add_bc("right(w) = 0", condition="(nx != 0) or (ny != 0)")
problem.add_bc("integ_z(p) = 0", condition="(nx == 0) and (ny == 0)")

# Build solver
solver = problem.build_solver(de.timesteppers.MCNAB2)
logger.info('Solver built')

#%%
# Initial conditions
z = domain.grid(2)
b = solver.state['b']
bz = solver.state['bz']

# Random perturbations, initialized globally for same results in parallel
gshape = domain.dist.grid_layout.global_shape(scales=1)
slices = domain.dist.grid_layout.slices(scales=1)
rand = np.random.RandomState(seed=23)
noise = rand.standard_normal(gshape)[slices]

# Linear background + perturbations damped at walls
zb, zt = z_basis.interval
pert =  1e-7* noise
b['g'] = N2*z + pert
b.differentiate('z', out=bz)
#%%
# Integration parameters
solver.stop_sim_time = 3600*24*10
solver.stop_wall_time = 60 * 60.
solver.stop_iteration = np.inf

# Analysis
snap = solver.evaluator.add_file_handler('snapshots', sim_dt=3600, max_writes=24)
snap.add_task("interp(p, y=16)", scales=1, name='p midplane')
snap.add_task("interp(b, y=16)", scales=1, name='b midplane')
snap.add_task("interp(u, y=16)", scales=1, name='u midplane')
snap.add_task("interp(v, y=16)", scales=1, name='v midplane')
snap.add_task("interp(w, y=16)", scales=1, name='w midplane')
snap.add_task("interp(b, z=100)", scales=1, name='b surface')

#snap.add_task("integ(b, 'z')", name='b integral x4', scales=4)

# CFL
CFL = flow_tools.CFL(solver, initial_dt=1e-4, cadence=1, safety=1.5,
                     max_change=1.5, min_change=0, max_dt=30)
CFL.add_velocities(('u', 'v', 'w'))

# Flow properties
#flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
#flow.add_property("sqrt(u*u + v*v + w*w) / R", name='Re')

# Main loop
end_init_time = time.time()
logger.info('Initialization time: %f' %(end_init_time-start_init_time))
try:
    logger.info('Starting loop')
    start_run_time = time.time()
    while solver.ok:
        dt = CFL.compute_dt()
        solver.step(dt)
        if (solver.iteration-1) % 100 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
#            logger.info('Max Re = %f' %flow.max('Re'))
            utemp = solver.state['u']
            logger.info('U Val: %f' %(utemp['g'][1, 1, 1]))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_run_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_run_time-start_run_time))
    logger.info('Run time: %f cpu-hr' %((end_run_time-start_run_time)/60/60*domain.dist.comm_cart.size))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 10:24:38 2017

@author: jacob
"""

"""
Dedalus script for Finite Amplitude Calcs

To run, merge, and plot using 4 processes, for instance, you could use:
    $ mpiexec -n 4 python3 rayleigh_benard.py
    $ mpiexec -n 4 python3 merge.py snapshots
    $ mpiexec -n 4 python3 plot_2d_series.py snapshots/*.h5

"""

import numpy as np
from mpi4py import MPI
CW = MPI.COMM_WORLD
import time
from pylab import *
from dedalus import public as de
from dedalus.extras import flow_tools
import scipy.integrate as integrate
import logging
logger = logging.getLogger(__name__)


# Parameters
#directoryname = '/scratch/jacob13/NLSIM/'
#ly_global = np.logspace(-5, -3, 192)*2*np.pi
#OD = False


nx = 64
ny = 64
Lx, Ly, Lz = (nx*1e3/2,ny*1e3/2, 1000.)

f = 1e-4 # Coriolis parameter
#N2 = (12*f)**2
#N = (3.4e-3)
tht = 0.01 # slope angle
kap4 = 2e5
nu = 1e-5
kap = nu # Unit Pr

BLH = 100
VINT = 0.1
Shmag = VINT/BLH
Ri = 1.5
N = np.sqrt(Ri*Shmag**2+f*Shmag/np.sin(tht)) # Defines a constant uniform background vertical gradient (also strength of across-slope gradient)

nz = 256

# CREATE A 1D DOMAIN FOR SIMPLICITY (this is a kludge)
z_basis = de.Chebyshev('z', nz, interval=(0, Lz), dealias=3/2)
domain1D = de.Domain([z_basis], grid_dtype=np.float64, comm=MPI.COMM_SELF)
z1 = domain1D.grid(0)

#%% 3D PROBLEM
# Create basis and domain
start_init_time = time.time()
x_basis = de.Fourier('x', nx, interval=(0, Lx), dealias=3/2)
y_basis = de.Fourier('y', ny, interval=(0, Ly), dealias=3/2)
z_basis = de.Chebyshev('z', nz, interval=(0, Lz), dealias=3/2)


domain = de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.float64, mesh=None)
z = domain.grid(2)
x = domain.grid(0)



# Define Fields

VI = domain.new_field(name='VI')
Vz = domain.new_field(name='Vz')
BZI = domain.new_field(name='BZI')
B = domain.new_field(name='B')
Vz['g'] = 0*z1
#Bt = np.zeros([nz])
Bzbl = -f*Shmag/np.sin(tht)

# Make lower BBL
# XXX - Am I defining these terms in z or zhat?
zind = np.floor( next((x[0] for x in enumerate(z1) if x[1]>BLH)))
tpoint = np.floor( next((x[0] for x in enumerate(z1) if x[1]>BLH)))
tpoint = int(tpoint)
BZI['g'][:,:,0:tpoint] = Bzbl # Set lower BL Bz negative to offset interior strat.
BZI['g'][:,:,0] = -N**2*np.cos(tht) # To satisfy lower BC   
Vz['g'][:,:,0:tpoint] = Shmag # Vertical shear only in BBL
#Bt[1:nz] = integrate.cumtrapz(BZI['g'], z)
#B['g'] = Bt

VI['g'][:,:,1:nz] = integrate.cumtrapz(Vz['g'], z1)
VI['g'][:,:,0] = 0 # Satisfy no-slip
# XXX - Do I need to do this?
VI['g'] = VI['g']-VINT #Imposing VINT separately to eliminate  pressure gradient terms
    

# set up IVP
problem = de.IVP(domain, variables=['u', 'v', 'w', 'b', 'p', 'uz', 'vz', 'wz', 'bz'])
problem.meta[:]['z']['dirichlet'] = True

slices = domain.dist.grid_layout.slices(scales=1)
z_slice = slices[2]

# define constants
problem.parameters['N'] = N #Fixed background stratification
problem.parameters['f'] = f
problem.parameters['tht'] = tht
problem.parameters['kap'] = kap
problem.parameters['L'] = Lx
problem.parameters['H'] = Lz
problem.parameters['A4'] = kap4 
problem.parameters['VI'] = VI 
problem.parameters['VIb'] = VINT
problem.parameters['BZI'] = BZI

# define substitutions
problem.substitutions['D(A,Az)'] = 'kap*dz(Az) + dz(kap)*Az' # Vertical diffusion operator
problem.substitutions['NL(A,Az)'] = 'u*dx(A) + v*dy(A) + w*Az' # nonlinear operator
problem.substitutions['HV(A)'] = '-A4*(dx(dx(dx(dx(A)))) + 2*dx(dx(dy(dy(A)))) + dy(dy(dy(dy(A)))))' #Horizontal biharmonic diff

# substitutions for diagnostics
problem.substitutions['zf'] = 'x*sin(tht) + z*cos(tht)'
problem.substitutions['havg(A)'] = "integ(A, 'x', 'y')/L**2"
problem.substitutions['prime(A)'] = "A - havg(A)"
problem.substitutions['EKE']  = '(prime(u)**2 + prime(v)**2)/2'
problem.substitutions['avg(A)'] = "integ(A, 'x', 'y', 'z')/L**2"
problem.substitutions['vint(A)'] = "integ(A, 'z')"

# define equations
problem.add_equation('dt(u) - f*v*cos(tht) + dx(p) - b*sin(tht) - D(u,uz) - HV(u) + VIb*dy(u)  = -NL(u,uz) - VI*dy(u)')
problem.add_equation('dt(v) + f*u*cos(tht) + dy(p) - D(v,vz) - HV(v)  + VIb*dy(v) = -NL(v,vz) - VI*dy(v) - w*dz(VI)+ D(VI,dz(VI))')
problem.add_equation('dz(p) - b*cos(tht) = 0')
#problem.add_equation('dt(w) + dz(p) - b*cos(tht) -D(w, wz) - HV(w) = -NL(w, wz) - VI*dy(w)')

problem.add_equation('dt(b) + u*N**2*sin(tht) + w*N**2*cos(tht) - D(b,bz) - HV(b) +VIb*dy(b) = -NL(b,bz) - VI*dy(b) - w*BZI +D(0, BZI+N**2*cos(tht))')

problem.add_equation('dx(u) + dy(v) + wz = 0')

# define derivatives
problem.add_equation('uz - dz(u) = 0')
problem.add_equation('vz - dz(v) = 0')
problem.add_equation('wz - dz(w) = 0')
problem.add_equation('bz - dz(b) = 0')

# define boundary conditions
problem.add_bc('left(u) = 0')
problem.add_bc('left(v) = 0')
problem.add_bc('left(w) = 0')
problem.add_bc('left(bz) = 0')
problem.add_bc('right(uz) = 0')
problem.add_bc('right(vz) = 0')
problem.add_bc('right(w) = 0', condition='(nx != 0) or (ny != 0)')
problem.add_bc('left(p) = 0', condition='(nx == 0) and (ny == 0)')
problem.add_bc('right(bz) = 0')

# Build solver
solver = problem.build_solver(de.timesteppers.MCNAB2)
logger.info('Solver built')

# Initial conditions
b = solver.state['b']
u = solver.state['u']
v = solver.state['v']
bz = solver.state['bz']
uz = solver.state['uz']
vz = solver.state['vz']

#%%
# define initial condtions
#b['g']= B['g']
#u['g']= U['g']
#v['g']= V['g']-VINT

b['g'] = 0
u['g'] = 0
v['g'] = 0

# Random perturbations, initialized globally for same results in parallel
gshape = domain.dist.grid_layout.global_shape(scales=1)
slices = domain.dist.grid_layout.slices(scales=1)
rand = np.random.RandomState(seed=23)
noise = rand.standard_normal(gshape)[slices]

b['g'] += nx*1e-10*noise

# Calculate Derivatives
b.differentiate('z', out=bz)
u.differentiate('z', out=uz)
v.differentiate('z', out=vz)

#%%
# Integration parameters
solver.stop_sim_time = 3600*24*20
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf

# Analysis
snap = solver.evaluator.add_file_handler('snapshots', sim_dt=3600*6, max_writes=24*1000, parallel=False)

# KE and PE
snap.add_task("-avg(zf*b)", name='pe')
snap.add_task("avg(u**2 + v**2)/2", name='ke')
snap.add_task("vint(havg(u)**2 + havg(v)**2)/2", name='mke')
snap.add_task('avg(EKE)', name='eke')

# buoyancy fields
snap.add_task("interp(b, z=0)", scales=1, name='b surface')
snap.add_task("interp(b, z=10)", scales=1, name='b 10')
snap.add_task("interp(b, z=25)", scales=1, name='b 25')
snap.add_task("interp(b, z=50)", scales=1, name='b 50')
snap.add_task("interp(b, z=100)", scales=1, name='b 100')
snap.add_task("interp(b, y=0)", scales=1, name='b plane')
# velocity field
snap.add_task("u", scales=1, name='u')
snap.add_task("v", scales=1, name='v')
snap.add_task("w", scales=1, name='w')
snap.add_task("b", scales=1, name='b')
snap.add_task('VI', scales=1, name='V')
snap.add_task('BZI', scales=1, name='Bz')
#snap.add_task('prime(u)', scales=1, name='u prime')
#snap.add_task('prime(v)', scales=1, name='v prime')

# KE budget
#snap.add_task("avg(prime(u)*prime(b)*sin(tht) + prime(w)*prime(b)*cos(tht))", name='byncy prdctn')
#snap.add_task("vint(havg(u)*havg(b)*sin(tht))", name='mean byncy prdctn')
#snap.add_task("-vint(kap*(havg(uz)**2 + havg(vz)**2))", name='mean dssptn')
#snap.add_task("vint(havg(uz)*havg(u*w) + havg(vz)*havg(v*w))", name='shear prod')

snap.add_task("havg(prime(w)*prime(b)*cos(tht) + prime(u)*prime(b)*sin(tht))", name='wpbp')
#snap.add_task("-havg(prime(v)*prime(w)*havg(vz))", name='shear prod')
snap.add_task("-havg(prime(v)*prime(w)*havg(dz(v+VI)) + prime(u)*prime(w)*havg(uz))", name='sp')
snap.add_task("havg(prime(u)*D(prime(u), prime(uz)) + prime(v)*D(prime(v+VI), prime(vz + dz(VI))))", name='vertical diss')
snap.add_task("havg(prime(u)*HV(prime(u)) + prime(v)*HV(prime(v+VI)))", name='hyper diss')

#snap.add_task("-havg(prime(v)*prime(w)*cos(tht) + prime(u)*prime(v)*sin(tht))*dz(VI)", name='mf shear prod')
#snap.add_task("-havg(prime(u)*prime(u)*dx(u) +prime(u)*prime(v)*dy(v) + prime(v)*prime(v)*dy(v) + prime(v)*prime(v)*dx(u))", name='lat shear prod')
#snap.add_task("-havg(prime(u)*u*dx(u) +prime(u)*(v+VI+VIb)*dy(v) + prime(v)*(v+VI+VIb)*dy(v) + prime(v)*(v+VI+VIb)*dx(u))", name='lat shear prod 2')
#snap.add_task("havg(prime(u)*prime(D(u,uz)) + prime(v)*prime(D(v,vz)))", name='dssptn')
#snap.add_task("havg(prime(v)*D(VI, dz(VI)))", name='mf dssptn')
#snap.add_task("havg(prime(u)*HV(u) +prime(v)*HV(v))", name='hypv')
#snap.add_task("-havg(prime(v)*prime(w)*dz(v+VI))", name='tot shear prod')
#snap.add_task("-havg(prime(v)*prime(w)*havg(dz(v+VI)))", name='avg shear prod')
#snap.add_task("havg(u*D(u,uz) + v*D(v,vz))", name='f dssptn')
#snap.add_task("havg((w)*(b))", name='f wpbp')

# CFL
CFL = flow_tools.CFL(solver, initial_dt=1e-4, cadence=1, safety=1.5,
                     max_change=1.5, min_change=0, max_dt=7200)
CFL.add_velocities(('u', 'v', 'w'))

# Flow properties

# Main loop
end_init_time = time.time()
logger.info('Initialization time: %f' %(end_init_time-start_init_time))
try:
    logger.info('Starting loop')
    start_run_time = time.time()
    while solver.ok:
        dt = CFL.compute_dt()
        solver.step(dt)
        if (solver.iteration-1) % 10 == 1:
            logger.info('Iteration: %i, Days: %1.1f, dt: %e' %(solver.iteration, solver.sim_time/86400, dt))
            utemp = solver.state['u']
            if utemp['g'].size > 0:
                um = np.max(np.abs(utemp['g']))
                logger.info('U Val: %f' % um)
                if np.isnan(um):
                    raise Exception('NaN encountered.')
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_run_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_run_time-start_run_time))
    logger.info('Run time: %f cpu-hr' %((end_run_time-start_run_time)/60/60*domain.dist.comm_cart.size))

import numpy as np
import matplotlib.pyplot as plt
from pylab import *

from dedalus import public as de
import time
from dedalus.extras import flow_tools
import logging
logger = logging.getLogger(__name__)

dt = 1e-1
# parameters
N = 3.5e-3 # buoyancy frequency
f = 1e-4 # Coriolis parameter
tht = 1e-2 # slope angle
kap_0 = 1e-4 # background diffusivity
kap_1 = 1e-3 # bottom enhancement of diffusivity
h = 200. # decay scale of mixing
Pr = 1 # Prandtl number
H = 1000. # domain height

# number of grid points
nz = 16


# file name that results are saved in
name = '/home/jacob/Dropbox/Slope BI/Figures/Ekman/'
directoryname = '/home/jacob/Dropbox/Slope BI/EkmanFiles/'
# build domain
z_basis = de.Chebyshev('z', nz, interval=(0, H), dealias=3/2)
domain = de.Domain([z_basis], np.float64)

# non-constant coefficients
kap = domain.new_field(name='kap')
z = domain.grid(0)
kap['g'] = kap_0/10*np.ones((nz, ))# + kap_1*np.exp(-z/h)
V = 0.1

# STEADY STATE

# setup problem
problem = de.IVP(domain, variables=['u','v',  'b', 'uz', 'vz', 'bz'])
problem.parameters['N'] = N
problem.parameters['V'] = V
problem.parameters['f'] = f
problem.parameters['tht'] = tht
problem.parameters['kap'] = kap
problem.parameters['kb'] = kap_0
#problem.substitutions['Ri'] = '(bz + N**2)/(vz**2 + uz**2 + 1e-20)'
#problem.substitutions['kap'] = '1e-2*0.5*(1 - tanh( (Ri - 0.25)/0.25))'
problem.add_equation(('dt(u) - f*v - b*sin(tht) + (- kb*dz(uz)) = + 0*f*V*cos(tht)- (- dz(kap)*uz - kap*dz(uz))'))
problem.add_equation(('dt(v) + f*u*cos(tht) + (- kb*dz(vz)) =  + 0 - (- dz(kap)*vz - kap*dz(vz))'))
problem.add_equation(('dt(b) + u*N**2*sin(tht) + (- kb*dz(bz)) '
        '= dz(kap)*N**2*cos(tht) - (- dz(kap)*bz - kap*dz(bz))'))
problem.add_equation('uz - dz(u) = 0')
problem.add_equation('vz - dz(v) = 0')
problem.add_equation('bz - dz(b) = 0')
problem.add_bc('left(u) = 0')
problem.add_bc('left(v) = -V')
problem.add_bc('left(bz) = -N**2*cos(tht)')
#problem.add_bc('right(uz) = 0')
#problem.add_bc('right(vz) = 0')
#problem.add_bc('right(bz) = 0')
problem.add_bc('right(u) = 0')
problem.add_bc('right(v) = 0')
problem.add_bc('right(b) = 0')

#problem.substitutions['Ri'] = '(bz + N**2)/(vz**2 + uz**2 + 1e-10)'
#problem.substitutions['kap'] = '1e-1*0.5*(1 - tanh( (Ri - 0.25)/0.25)) + 1e-5'
#problem.add_equation(('dt(u) - f*v - b*sin(tht)   = + 0*f*V*cos(tht) - (- 0*dz(kap)*uz - kap*dz(uz))'))
#problem.add_equation(('dt(v) + f*u*cos(tht)  =  -(- 0*dz(kap)*vz - kap*dz(vz))'))
#problem.add_equation(('dt(b) + u*N**2*sin(tht)  '
#        '= dz(kap)*N**2*cos(tht) -  (- 0*dz(kap)*bz - kap*dz(bz))'))
#problem.add_equation('uz - dz(u) = 0')
#problem.add_equation('vz - dz(v) = 0')
#problem.add_equation('bz - dz(b) = 0')
#problem.add_bc('left(u) = 0')
#problem.add_bc('left(v) = -V')
#problem.add_bc('left(bz) = -N**2*cos(tht)')
#problem.add_bc('right(uz) = 0')
#problem.add_bc('right(vz) = 0')
#problem.add_bc('right(bz) = 0')

# build solver and solve
# time stepping
ts = de.timesteppers.RK443
solver = problem.build_solver(ts)

u = solver.state['u']
v = solver.state['v']
b = solver.state['b']
bz = solver.state['bz']
uz = solver.state['uz']
vz = solver.state['vz']

# initial conditions
cek = -V*np.exp(-z*np.sqrt(-1j*2*kap_0/f))
u['g'] = 0*cek.imag
v['g'] = 0*cek.real
v['g'][0] = -V
b['g'] = 0


# flow control
solver.stop_sim_time = 86400*5
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf

# plot setup
z = domain.grid(0, scales=1)
zd = domain.grid(0, scales=domain.dealias)
logger.info('Starting loop')
start_time = time.time()

#CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=1, safety=1.5,
#                     max_change=1.5, min_change=0, max_dt=1e-1)
#CFL.add_velocities(('u'))

while solver.ok:

    # time step
    solver.step(dt)
#    dt = CFL.compute_dt()
    # Turbulence parameterization
    Ri = (bz['g'] + N**2*np.cos(tht))/(vz['g']**2 + uz['g']**2 +1e-20)
    
    # Possible smooth Ri function
#    kap['g'] = 0*1e-2*0.5*(1 - np.tanh( (Ri - 0.3)/0.3)) 
    
    # Following Benthuysen and Thomas 2012
    maxv = 1e-2
    minv = 0*kap_0/10
#    kapold = kap['g']
#    kap['g']         = (-maxv + 0)*10*(Ri-0.2)+maxv
#    kap['g'][Ri>0.3] = 0
#    kap['g'][Ri<0.2] = maxv
#    kap['g'] = 1/2*(kapold + kap['g'])
    
#    problem.parameters['kap']['g'] = kap
    
    if solver.iteration*dt % 60 == 0:
        logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
        logger.info('Max u: %f  Max v: %f' % (np.max(np.abs(u['g'])), np.max(np.abs(v['g']))) )
        
    if solver.iteration*dt % 1 == 0:
        logger.info('Modifying Viscosity')
        kap['g']         = (-maxv + minv)*10*(Ri-0.2)+maxv
        kap['g'][Ri>0.3] = minv
        kap['g'][Ri<0.2] = maxv
        
        kap['g'] = maxv
        problem.parameters['kap']['g'] = kap['g']
#        simtime = solver.sim_time
#        iteration = solver.iteration
#        stoptime = solver.stop_sim_time
#        solver = problem.build_solver(ts)
#        solver.sim_time = simtime
#        solver.iteration = iteration
#        solver.stop_sim_time = stoptime
#        solver.stop_iteration = np.inf
    if solver.iteration*dt % 1800 == 0:
        logger.info('Saving File')
        # Make plot of u at surface
        fig, ax = plt.subplots(1, 3, sharey=True)
        ax[0].plot( v['g']+V, z)
        ax[0].plot( u['g'], z)  
        ax[1].plot(b['g'] + N**2*z, z)
        plt.title("Hour: %.2f" %(solver.sim_time/3600))
        ax[2].semilogx(kap['g']+kap_0, zd)
        
#    if solver.iteration > 100:
#        dt = 5e-1
#        plt.colorbar()
        fig.savefig(name + 'ekfig_{:010d}.png'.format(np.int(solver.sim_time)), dpi=300)
        plt.close()
        
        np.savez(directoryname+'ekfile_{:010d}.npz'.format(np.int(solver.sim_time)), nz=nz, tht=tht, z=z, f=f, kap=kap['g']+kap_0,
        V=V, b=b['g'], bz=bz['g'], v = v['g'], vz=vz['g'],u=u['g'], uz=uz['g'], N = N,  H = H)
        # Make plot of v at surface
#        plt.figure()
#        plt.pcolormesh(x[:,0,0], y[0,:,0], v['g'][:,:,-1].T, Rasterized=True)
#        plt.colorbar()
#        plt.savefig(name + '/v_surf_{:010d}.png'.format(solver.iteration), dpi=300)
#        plt.close()
#        # Make plot of b at surface
#        plt.figure()
#        plt.pcolormesh(x[:,0,0], y[0,:,0], (-lmd*(y-.5) + b['g'])[:,:,-1].T, rasterized=True)
#        plt.colorbar()
#        plt.clim(-lmd, lmd)
#        plt.savefig(name + '/b_surf_{:010d}.png'.format(solver.iteration), dpi=300)
#        plt.close()

end_time = time.time()

# Print statistics
logger.info('Run time: %f' %(end_time-start_time))
logger.info('Iterations: %i' %solver.iteration)


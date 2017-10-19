import numpy as np
import matplotlib.pyplot as plt
from pylab import *

from dedalus import public as de
import time
from dedalus.extras import flow_tools
import logging
logger = logging.getLogger(__name__)

dt = 1e-0
# parameters
N = 3.5e-3 # buoyancy frequency
f = 1e-4 # Coriolis parameter
tht = 1e-2 # slope angle
kap_0 = 1e-4 # background diffusivity
kap_1 = 1e-3 # bottom enhancement of diffusivity
h = 200. # decay scale of mixing
Pr = 1 # Prandtl number
H = 100. # domain height
V = .1

# number of grid points
nz = 500

# file name that results are saved in
name = '/home/jacob/Dropbox/Slope BI/Figures/Ekman/'
directoryname = '/home/jacob/Dropbox/Slope BI/EkmanFiles/'
# build domain
z = np.linspace(0, H, nz)
dz = z[1]-z[0]
# STEADY STATE
# initial conditions


u = np.zeros(z.shape)
v = np.zeros(z.shape)
b = np.zeros(z.shape)

v[0] = -V
kap = kap_0*np.ones(z.shape)
    
stoptime = 86400*25

#CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=1, safety=1.5,
#                     max_change=1.5, min_change=0, max_dt=1e-1)
#CFL.add_velocities(('u'))
iteration =  0

vfull = np.zeros((nz+2,))
vfull[0] = -V
ufull = np.zeros((nz+2,))
bfull = np.zeros((nz+2,))
lbc = -N**2*np.cos(tht)
kapfull = np.zeros((nz+2,))

def dif1_matrix(x):
    '''
    In case for len(x) = 5:
    dif = np.array(
    [[1, 0, 0, 0, 0],
     [-1, 1, 0, 0, 0],
     [0, -1, 1, 0, 0],
     [0, 0, -1, 1, 0],
     [0, 0, 0, -1, 1]])
    '''
    # Diagonal elements are 1.
    dif_now = np.diag(np.ones(len(x)+2))
 
    # Left elements of diagonal are -1.
    dif_pre_ones = np.ones(len(x))  # -1 vector.
#        dif_pre = np.diag(dif_now, k=0) # Diagonal matrix shiftedto left.
    dif_pre = dif_now[0:-2,:]
    dif_post = np.diag(dif_pre_ones, k=2)
    dif_post = dif_post[0:-2,:]
    dif = (dif_post - dif_pre)/(2*dz)
    return dif
    
def dif2_matrix(x):
    '''
    In case for len(x) = 5:
    dif = np.array(
    [[1, 0, 0, 0, 0],
     [-1, 1, 0, 0, 0],
     [0, -1, 1, 0, 0],
     [0, 0, -1, 1, 0],
     [0, 0, 0, -1, 1]])
    '''
    # Diagonal elements are 1.
    dif_now = np.diag(np.ones(len(x)+2))
 
    # Left elements of diagonal are -1.
    dif_pre_ones = np.ones(len(x))  # -1 vector.
#        dif_pre = np.diag(dif_now, k=0) # Diagonal matrix shiftedto left.
    dif_pre = dif_now[0:-2,:]
    dif_post = np.diag(dif_pre_ones, k=2)
    dif_post = dif_post[0:-2,:]
    dif_now = np.diag(np.ones(len(x)+1), k=1)
    dif_now = dif_now[0:-2,:]
    dif = (dif_post + dif_pre - 2*dif_now)/(dz**2)
    return dif

deriv = dif1_matrix(u)
deriv2 = dif2_matrix(u)
    
    
while iteration < stoptime:
    uold = u
    vold = v
    bold = b
    
    vfull[1:-1] = vold
    ufull[1:-1] = uold
    bfull[1:-1] = bold
    bfull[0] = -2*dz*lbc + b[1]
    

    #Following Benthuysen and Thomas 2012
    Ri = (deriv.dot(bfull) + N**2*np.cos(tht))/(deriv.dot(vfull)**2 + deriv.dot(ufull)**2 +1e-20)

    maxv = 1e-2
    minv = kap_0
    kap         = (-maxv + minv)*10*(Ri-0.2)+maxv
    kap[Ri>0.3] = minv
    kap[Ri<0.2] = maxv
    kapfull[1:-1] = kap
    # Convert ends to forward differences
#    
#    deriv[0,0] = -1/dz
#    deriv[0,1] = 1/dz
#    deriv[-1, -1] = 1/dz
#    deriv[-1, -2] = -1/dz
#    
#    vderiv = deriv
#    vderiv[0,0] = vderiv[0,0]*-V
#    vderiv[-1,-1] = 0
#    
#    uderiv = 
#    
#    derivv = deriv
#    derivv[0,0] = -V*2
#    derivv[0,1] = derivv[0,1]*2
    
    
    u = uold + dt*(f*vold+bold*np.sin(tht)+deriv.dot(kapfull)*deriv.dot(ufull) + kap*deriv2.dot(ufull))
    v = vold + dt*(-f*uold*np.cos(tht) + deriv.dot(kapfull)*deriv.dot(vfull) + kap*deriv2.dot(vfull))
    b = bold + dt*(-uold*N**2*np.sin(tht) + deriv.dot(kapfull)*(N**2*np.cos(tht)+ deriv.dot(bfull)) + kap*deriv2.dot(bfull))



#    u = uold + dt*(f*vold+bold*np.sin(tht)+np.gradient(kap, dz)*np.gradient(uold, dz) + kap*np.gradient(np.gradient(uold,dz), dz))
#    v = vold + dt*(-f*uold*np.cos(tht) + np.gradient(kap, dz)*np.gradient(vold, dz) + kap*np.gradient(np.gradient(vold,dz), dz))
#    b = bold + dt*(-uold*N**2*np.sin(tht) + np.gradient(kap, dz)*np.gradient(bold, dz) + kap*np.gradient(np.gradient(bold,dz), dz))
    # time step
    iteration = iteration + 1
#    dt = CFL.compute_dt()
    # Turbulence parameterization
    
    # Possible smooth Ri function
#    kap['g'] = 0*1e-2*0.5*(1 - np.tanh( (Ri - 0.3)/0.3)) 
    

    
#    problem.parameters['kap']['g'] = kap
    
    if iteration*dt % 60 == 0:
        logger.info('Iteration: %i, Time: %e, dt: %e' %(iteration, iteration*dt, dt))
        logger.info('Max u: %f  Max v: %f' % (np.max(np.abs(u)), np.max(np.abs(v))) )
        
#    if solver.iteration*dt % 1 == 0:
#        logger.info('Modifying Viscosity')
#        kap['g']         = (-maxv + minv)*10*(Ri-0.2)+maxv
#        kap['g'][Ri>0.3] = minv
#        kap['g'][Ri<0.2] = maxv
#        
#        kap['g'] = maxv
#        problem.parameters['kap']['g'] = kap['g']
#        simtime = solver.sim_time
#        iteration = solver.iteration
#        stoptime = solver.stop_sim_time
#        solver = problem.build_solver(ts)
#        solver.sim_time = simtime
#        solver.iteration = iteration
#        solver.stop_sim_time = stoptime
#        solver.stop_iteration = np.inf
    if iteration*dt % 21600 == 0:
        logger.info('Saving File')
        # Make plot of u at surface
        fig, ax = plt.subplots(1, 3, sharey=True)
        ax[0].plot( v+V, z)
        ax[0].plot( u, z)  
        ax[0].set_xlabel('U and V')
        ax[0].set_ylabel('z (m)')
        ax[1].plot(b + N**2*z, z)
        ax[1].set_xlabel('b')
        ax[1].set_title("Hour: %.2f" %(iteration*dt/3600))
        ax[2].semilogx(kap+kap_0, z)
        ax[2].set_xlabel('nu ($m^2s^{-1}$)')
#    if solver.iteration > 100:
#        dt = 5e-1
#        plt.colorbar()

        
        fig.savefig(name + 'ekfig_{:010d}.png'.format(np.int(iteration)), dpi=300)
        plt.close()
        
        # Save files
        uold = u
        vold = v
        bold = b
        vfull[1:-1] = vold
        ufull[1:-1] = uold
        bfull[1:-1] = bold
        bfull[0] = -2*dz*lbc + b[1]
        
        np.savez(directoryname+'ekfile_{:010d}.npz'.format(np.int(iteration*dt)), tht=tht, z=z, f=f, kap=kap,
        V=V, b=b, bz=deriv.dot(bfull), v = v, vz=deriv.dot(vfull),u=u, uz=deriv.dot(ufull), N = N,  H = H, time=np.int(iteration*dt))
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
#logger.info('Run time: %f' %(end_time-start_time))
#logger.info('Iterations: %i' %solver.iteration)


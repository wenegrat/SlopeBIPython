import numpy as np
import matplotlib.pyplot as plt

from dedalus import public as de
import scipy.integrate as integrate

import logging
logger = logging.getLogger(__name__)

# parameters
N = 1e-4 # buoyancy frequency
Pr = 0.1
f = Pr*N # Coriolis parameter
#tht = -5e-2 # slope angle
H = 100. # domain height
Ri =1.25 # CI PARAMETERS
S = 2.75

#Ri = 0.5  #SI PARAMETERS
#S = 1

directoryname = '/home/jacob/dedalus/SIRegimeScan2/'

# slope parameter
#print(np.tan(tht)*N**2/(f*Lmd))

# along-slope wavenumbers
ll = np.linspace(.01*f/(N*H), 20*f/(N*H), 2)

# number of grid points
nz = 128

rivec = np.linspace(1e-2, 5, 64)
svec = np.linspace(0, 2, 64)


# build domain
z_basis = de.Chebyshev('z', nz, interval=(0, H))
domain = de.Domain([z_basis], np.complex128)

# LINEAR STABILITY
for Ri in rivec:
    for S in svec:
        Lmd = N/Ri**(1/2) # shear
        tht = np.arctan(f/N*S**(1/2))
        print('Ri: ' + str(Ri) + '  S: ' + str(S))
        
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
        problem.add_equation('dt(u) + Lmd*z/cos(tht)*dy(u) - f*v*cos(tht) + dx(p) - b*sin(tht)= 0')
        problem.add_equation('dt(v) + Lmd*z/cos(tht)*dy(v) + w*Lmd/cos(tht) + f*u*cos(tht) - f*w*sin(tht) + dy(p) = 0')
        problem.add_equation('(dt(w) + Lmd*z/cos(tht)*dy(w)) + f*v*sin(tht) + dz(p) - b*cos(tht) = 0')
        problem.add_equation('dt(b) + Lmd*z/cos(tht)*dy(b) + u*(N**2*sin(tht) + f*Lmd*cos(tht)) + w*(N**2*cos(tht) - f*Lmd*sin(tht)) = 0')
        problem.add_equation('dx(u) + dy(v) + dz(w) = 0')
        problem.add_bc('left(w) = 0')
        problem.add_bc('right(w) = 0')

#   set up solver
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
        
        
            # solve eigenvalue problem and sort
            idx = sorted_eigen(k, l)
        
            return solver.eigenvalues[idx[-1]].imag

        # get max growth rates
        gr = np.array([max_growth_rate(l, 0) for l in ll])
        
        # get full eigenvectors and eigenvalues for l with largest growth
        idx = sorted_eigen(ll[np.argmax(gr)], 0.)
        solver.set_state(idx[-1])
        grr = solver.eigenvalues[idx[-1]].real
        
        # collect eigenvector
        u = solver.state['u']
        v = solver.state['v']
        w = solver.state['w']
        b = solver.state['b']
        
        z = domain.grid(0)

        Vzhat = Lmd
        VSP = -2*np.real((w['g'])*np.conj(v['g'])*Vzhat*np.cos(tht) +0* np.conj(v['g'])*u['g']*Vzhat*np.sin(tht))
        Vx = -Lmd/np.cos(tht)*np.sin(tht)
        LSP = -2*np.real(0*np.conj(v['g'])*u['g']*np.cos(tht) -(w['g'])*np.conj(v['g'])*np.sin(tht))*Vx

        BP = 2*np.real((u['g']*np.sin(tht)+w['g']*np.cos(tht))*np.conj(b['g']))
        HHF = -2*np.real(f*Lmd/N**2*(u['g']*np.cos(tht)-w['g']*np.sin(tht))*np.conj(b['g']))

        name = 'StabilityData_'+str(Ri)+'_' + str(S) # Can vary this depending on parameter of interest
        np.savez(directoryname+name + '.npz', nz=nz, tht=tht, N = N, Lmd = Lmd, z=z, f=f, k = ll[np.argmax(gr)],
                                    VSP= VSP, LSP = LSP, BP = BP, HHF=HHF, gr = np.max(gr), grr = grr, S=S, Ri = Ri)



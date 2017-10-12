#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 09:55:48 2017

@author: jacob
Based on code from Joern Callies

"""
def linearStabilityAnalysis(f, tht, kappa, Pr, U, Uz, V, Vz, B, Bz, nz, H, ll, domain):
    # Need to decide whether input variables are in rotated frame or not.
    
    import numpy as np
    import pdb
#    import matplotlib.pyplot as plt
#    from pylab import *
#    import scipy.integrate as integrate
    from dedalus import public as de

#    import logging
#    logger = logging.getLogger(__name__)
    
    # LOAD IN PARAMETERS
    #f = parameters, tht, kappa, Pr, U, Uz, V, Vz, B, Bz, nz, H, ll, domain)
#    f, tht, kappa, Pr, U, Uz, V, Vz, B, Bz, nz, H, ll, domain):
#        pdb.set_trace()

#    f = args[0]
#    tht = args[1]
#    kappa = args[2]
#    Pr = args[3]
#    U = args[4]
#    Uz = args[5]
#    V = args[6]
#    Vz = args[7]
#    B = args[8]
#    Bz = args[9]
#    nz = args[10]
#    H = args[11]
#    ll = args[12]
##    domain = args[13]
#    z_basis = de.Chebyshev('z', nz, interval=(0, H))
#    domain = de.Domain([z_basis], np.complex128)
    # non-constant coefficients
#    kap = domain.new_field(name='kap')
#    z = domain.grid(0)
#    kap['g'] = kapi
    print(f)
    # SETUP LINEAR STABILITY ANALYSIS
    problem = de.EVP(domain, variables=['u', 'v', 'w', 'b', 'p', 'uz', 'vz', 'wz',
            'bz'], eigenvalue='omg')
    problem.parameters['U'] = U
    problem.parameters['V'] = V
    problem.parameters['B'] = B
    problem.parameters['Uz'] = Uz
    problem.parameters['Vz'] = Vz
    problem.parameters['Bz'] = Bz
    problem.parameters['f'] = f
    problem.parameters['tht'] = tht
    problem.parameters['kap'] = kappa
    problem.parameters['Pr'] = Pr
    problem.parameters['k'] = 0. # will be set in loop
    problem.parameters['l'] = 0. # will be set in loop
    problem.substitutions['dx(A)'] = "1j*k*A"
    problem.substitutions['dy(A)'] = "1j*l*A"
    problem.substitutions['dt(A)'] = "-1j*omg*A"
    problem.add_equation(('dt(u) + U*dx(u) + V*dy(u) + w*Uz - f*v*cos(tht) + dx(p)'
            '- b*sin(tht) - Pr*(kap*dx(dx(u)) + kap*dy(dy(u)) + dz(kap)*uz'
            '+ kap*dz(uz)) = 0'))
    problem.add_equation(('dt(v) + U*dx(v) + V*dy(v) + w*Vz + f*u*cos(tht)'
            '- f*w*sin(tht) + dy(p) - Pr*(kap*dx(dx(v)) + kap*dy(dy(v))'
            '+ dz(kap)*vz + kap*dz(vz)) = 0'))
    problem.add_equation(('dt(w) + U*dx(w) + V*dy(w) + f*v*sin(tht) + dz(p)'
            '- b*cos(tht) - Pr*(kap*dx(dx(w)) + kap*dy(dy(w)) + dz(kap)*wz'
            '+ kap*dz(wz)) = 0'))
#    problem.add_equation(('dt(b) + U*dx(b) + V*dy(b) + u*N**2*sin(tht)'
#            '+ w*(N**2*cos(tht) + Bz) - kap*dx(dx(b)) - kap*dy(dy(b)) - dz(kap)*bz'
#            '- kap*dz(bz) = 0'))
    problem.add_equation(('dt(b) + U*dx(b) + V*dy(b) + u*Vz*f'
            '+ w*(Bz) - kap*dx(dx(b)) - kap*dy(dy(b)) - dz(kap)*bz'
            '- kap*dz(bz) = 0'))
    problem.add_equation('dx(u) + dy(v) + wz = 0')
    problem.add_equation('uz - dz(u) = 0')
    problem.add_equation('vz - dz(v) = 0')
    problem.add_equation('wz - dz(w) = 0')
    problem.add_equation('bz - dz(b) = 0')
    problem.add_bc('left(u) = 0')
    problem.add_bc('left(v) = 0')
    problem.add_bc('left(w) = 0')
    problem.add_bc('left(bz) = 0')
    problem.add_bc('right(uz) = 0')
    problem.add_bc('right(vz) = 0')
    problem.add_bc('right(w) = 0')
    problem.add_bc('right(bz) = 0')
    
    
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

    #get full eigenvectors and eigenvalues for l with largest growth
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
    
    return (gr, u, v, w, b, SP, BP)
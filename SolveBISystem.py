#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 14:44:46 2017

@author: jacob
"""

from sympy import *

Ldy, Ldz = var('Ldy Ldz')
g, x, y, z = var('g x y z')
xZ, yZ, zZ = var('xZ yZ zZ')
xdd, ydd, zdd = var('xdd ydd zdd')

E1 = z * xdd + (xZ - x) * (g + zdd)
E2 = z * ydd + (yZ - y) * (g + zdd) - Ldy
E3 = -y * xdd + x * ydd - zZ * (g + zdd) + Ldz

sols = solve([E1, E2, E3], [xdd, ydd, Ldy])

#print "xdd = ", (sols[xdd]).factor()
#print "ydd = ", (sols[ydd]).factor()
#print "Ldy = ", (sols[Ldy]).factor()

#%%

gam, Vz = var('gam Vz') #(lV-omega)
u, v, p, pz, b = var('u v p pz b')
l = var('l')
Ri = var('Ri')
delta = var('delta')
w, wz, wzz= var('w wz wzz')

E1 = 1j*gam*u - v - delta*b
E2 = 1j*gam*v + w*Vz + u + 1j*Ri*l*p
E3 = delta*Ri**(-1)*v + pz - b
E4 = 1j*l*v + wz
E5 = 1j*gam*b + Ri**(-1)*u*(delta+1)+w*(1-delta*Ri**(-1))

solsv = solve([E4], [v]) # Use continuity to eliminate v

E1 = 1j*gam*u - solsv[v] - delta*b
E2 = 1j*gam*solsv[v] + w*Vz + u + 1j*Ri*l*p
E3 = delta*Ri**(-1)*solsv[v] + pz - b
E5 = 1j*gam*b + Ri**(-1)*u*(delta+1)+w*(1-delta*Ri**(-1))

solsu = solve([E1], [u]) # use zonal momentum for u

E2 = 1j*gam*solsv[v] + w*Vz + solsu[u] + 1j*Ri*l*p
E3 = delta*Ri**(-1)*solsv[v] + pz - b
E5 = 1j*gam*b + Ri**(-1)*solsu[u]*(delta+1)+w*(1-delta*Ri**(-1))

solsb = solve([E5], [b])

E1 = 1j*gam*u - solsv[v] - delta*solsb[b]
solsu = solve([E1], [u]) # resolve for b

E2 = 1j*gam*solsv[v] + w*Vz + solsu[u] + 1j*Ri*l*p
E3 = delta*Ri**(-1)*solsv[v] + pz - solsb[b]

solspz = solve([E3], [pz])
E6 = 1j*Ri*l*solspz[pz] + Vz*wz - gam*wzz/l + (-Ri*delta*l*wz - Ri*gam*wzz + delta**2*l*wz)/(l*(-Ri*gam**2 + delta**2 + delta))


#%%

gam, Vz = var('gam Vz') #(lV-omega)
u, v, p, pz, b = var('u v p pz b')
l = var('l')
Ri = var('Ri')
delta = var('delta')
S = var('S')
w, wz, wzz= var('w wz wzz')

E1 = 1j*gam*u - v - 0*delta*b
E2 = 1j*gam*v + w*Vz + u + 1j*Ri*l*p
E3 = S*v + pz - b
E4 = 1j*l*v + wz
E5 = 1j*gam*b + u*(S+Ri**(-1))+w*(1-S)

solsv = solve([E4], [v]) # Use continuity to eliminate v

E1 = 1j*gam*u - solsv[v] - 0*delta*b
E2 = 1j*gam*solsv[v] + w*Vz + u + 1j*Ri*l*p
E5 = 1j*gam*b + u*(S+Ri**(-1))+w*(1-S)

solsu = solve([E1], [u]) # use zonal momentum for u

E2 = 1j*gam*solsv[v] + w*Vz + solsu[u] + 1j*Ri*l*p
E3 = S*solsv[v] + pz - b
E5 = 1j*gam*b + solsu[u]*(S+Ri**(-1))+w*(1-S)

solsb = solve([E5], [b])

#E1 = 1j*gam*u - solsv[v] - 0*delta*solsb[b]
#solsu = solve([E1], [u]) # resolve for b

#E2 = 1j*gam*solsv[v] + w*Vz + solsu[u] + 1j*Ri*l*p
E3 = delta*Ri**(-1)*solsv[v] + pz - solsb[b]
#
solspz = solve([E3], [pz])
E6 = (1/gam - gam)*wzz - Vz*l/gam**(2)*wz + 1j*l**(2)*solspz[pz]
#E6 = 1j*Ri*l*solspz[pz] + Vz*wz - gam*wzz/l + (-Ri*delta*l*wz - Ri*gam*wzz + delta**2*l*wz)/(l*(-Ri*gam**2 + delta**2 + delta))
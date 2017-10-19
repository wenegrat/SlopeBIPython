#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 15:39:06 2017

@author: jacob
"""
import numpy as np

alpha = 5e-3
N = 8.7e-3
f = 0.9e-4
delta = 1
GRND = 0.1 # non-dim growth rate => grn = gr*N/M^2

S = (N/f)*alpha
Ri = (delta/S)**2
M2 = np.sqrt(N**2*f**2/Ri)

omegadim = M2/N * GRND
ts = 2*np.pi/omegadim

alphamax = 1.5*f/(N*Ri**(1/2))
print("S = "+str(S)+ "  Ri = "+ str(Ri)+ " Timescale = " + str(ts/(24*3600)))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 17:33:30 2017

Consider the dependence on gamma
@author: jacob
"""
import numpy as np
Vo = 0.1
No = 5e-3
f = 1e-4
theta = 5e-3
So = (No/f*np.tan(theta))**2
gc = (1+So)**(-1)

gammas = np.linspace(gc, -1, 100)


h = Vo*np.sin(theta)/(gammas*f*So)
h[h<0] = -h[h<0] # reverse for interior flow direction

plt.figure()
plt.plot(gammas, h)
plt.ylim((0, 1000))
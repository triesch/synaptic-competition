#!/usr/bin/env python

# Filling Fraction in the steady state
#
# Jochen Triesch, January 2017 to January 2018
#
# This program was used to create Fig. 2A of the manuscript.
# It plots the filling fraction F as a function of (beta delta)/(alpha gamma)
# It also plots alpha as a function of F (for fixed gamma, delta).

import numpy as np
import matplotlib.pyplot as plt

# Part I: plot F as a function of (beta delta)/(alpha gamma)
x = np.arange(0.0, 10.0, 0.01) # x represents (beta delta)/(alpha gamma)
F = 1.0/(1.0 + x) 
      
f = plt.figure(figsize=(4,3))
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 12}
plt.rc('font', **font)
plt.rc('font', serif='Times New Roman') 
plt.plot(x, F, 'k')
plt.xlabel(r'Removal Ratio $(\beta \delta)/(\alpha \gamma)$')
plt.ylabel(r'Filling Fraction $F$')
plt.show()
f.savefig("Fig2A.pdf", bbox_inches='tight')

# Part II: plot alpha as a function of F (for delta = gamma = 1)
F = np.arange(0.0, 1.0, 0.01)
alpha = F / (1.0 - F)

f = plt.figure(figsize=(4,3))
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 12}
plt.rc('font', **font)
plt.rc('font', serif='Times New Roman') 
plt.plot(F, alpha, 'k')
plt.xlabel(r'$F$')
plt.ylabel(r'$\alpha \; [\beta / p^\infty]$')
plt.show()
#f.savefig("alpha.pdf", bbox_inches='tight')

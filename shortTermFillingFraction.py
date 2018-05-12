#!/usr/bin/env python

# Simple model of receptors diffusing in and out of synapses.
# Plotting the filling fraction F as a function of alpha: F= F(alpha)
#
# Jochen Triesch, April 2017 - January 2018
#
# This program was used to create Fig. 3 of the manuscript.
# It plots the short-term filling fraction (under the assumption of separation
# of timescales) for different numbers of slots and receptors.

import numpy as np
import  matplotlib.pyplot as plt

rho_exp = np.arange(1, 8, 0.1)
rho=10**rho_exp

# part I: fixed slots and varying receptors
S = 10000.0
R1 = 2000.0
R2 = 10000.0
R3 = 50000.0

W1 = 0.5*(S+R1+rho)-np.sqrt(0.25*(S+R1+rho)**2-R1*S)
W2 = 0.5*(S+R2+rho)-np.sqrt(0.25*(S+R2+rho)**2-R2*S)
W3 = 0.5*(S+R3+rho)-np.sqrt(0.25*(S+R3+rho)**2-R3*S)

F1 = W1/S
F2 = W2/S
F3 = W3/S

font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 12}
plt.rc('font', **font)

f1 = plt.figure(figsize=(4,3))    
plt.plot(rho_exp, F3, 'r', label=r'$R = 50000$')
plt.plot(rho_exp, F2, 'k', label=r'$R = 10000$')
plt.plot(rho_exp, F1, 'b', label=r'$R = 2000$')
plt.title(r'$S=10000$', fontsize=12)
plt.xlabel(r'$\log(\rho)$', fontsize=12)
plt.ylabel(r'$F^*$', fontsize=12)
plt.legend(loc='upper right', fontsize=12)
plt.show()
#f1.savefig("Fig3A.pdf", bbox_inches='tight')

f2 = plt.figure(figsize=(4,3))    
plt.plot(rho, F3, 'r', label=r'$R = 50000$')
plt.plot(rho, F2, 'k', label=r'$R = 10000$')
plt.plot(rho, F1, 'b', label=r'$R = 2000$')
plt.axis((0.0, 50000.0, 0.0, 1.0))
plt.title(r'$S=10000$', fontsize=12)
plt.xlabel(r'$\rho$', fontsize=12)
plt.ylabel(r'$F^*$', fontsize=12)
plt.legend(loc='upper right', fontsize=12)
plt.show()
#f2.savefig("Fig3C.pdf", bbox_inches='tight')


# part II: fixed receptors and varying slots
R = 10000.0
S1 = 2000.0
S2 = 10000.0
S3 = 50000.0

W1 = 0.5*(S1+R+rho)-np.sqrt(0.25*(S1+R+rho)**2-R*S1)
W2 = 0.5*(S2+R+rho)-np.sqrt(0.25*(S2+R+rho)**2-R*S2)
W3 = 0.5*(S3+R+rho)-np.sqrt(0.25*(S3+R+rho)**2-R*S3)

F1 = W1/S1
F2 = W2/S2
F3 = W3/S3

f3 = plt.figure(figsize=(4,3))    
plt.plot(rho_exp, F1, 'b', label=r'$S = 2000$')
plt.plot(rho_exp, F2, 'k', label=r'$S = 10000$')
plt.plot(rho_exp, F3, 'r', label=r'$S = 50000$')
plt.title(r'$R=10000$', fontsize=12)
plt.xlabel(r'$\log(\rho)$', fontsize=12)
plt.ylabel(r'$F^*$', fontsize=12)
plt.legend(loc='upper right', fontsize=12)
plt.show()
#f3.savefig("Fig3B.pdf", bbox_inches='tight')

f4 = plt.figure(figsize=(4,3))    
plt.plot(rho, F1, 'b', label=r'$S = 2000$')
plt.plot(rho, F2, 'k', label=r'$S = 10000$')
plt.plot(rho, F3, 'r', label=r'$S = 50000$')
plt.axis((0.0, 50000.0, 0.0, 1.0))
plt.title(r'$R=10000$', fontsize=12)
plt.xlabel(r'$\rho$', fontsize=12)
plt.ylabel(r'$F^*$', fontsize=12)
plt.legend(loc='upper right', fontsize=12)
plt.show()
#f4.savefig("Fig3D.pdf", bbox_inches='tight')


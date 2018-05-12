#!/usr/bin/env python

# Short-term hetersynaptic effect due to change in total number of slots
#
# Jochen Triesch, January-July 2017

import numpy as np
import matplotlib.pyplot as plt

S = 100.0           # initial number of slots
beta = 60.0/43.0    # transition rate out of slots in 1/min
delta = 1.0/14.0    # removal rate in 1/min

F1 = 0.5            # desired filling fractions
F2 = 0.7
F3 = 0.9
phi = 2.67          # relative pool size

P1 = phi*F1*S       # receptors in the pool in steady state
P2 = phi*F2*S
P3 = phi*F3*S

alpha1 = beta/(phi*S*(1-F1)) # set alphas accordingly
alpha2 = beta/(phi*S*(1-F2))
alpha3 = beta/(phi*S*(1-F3))

rho1 = beta/alpha1
rho2 = beta/alpha2
rho3 = beta/alpha3

R1 = P1 + F1*S         # total number of receptors
R2 = P2 + F2*S
R3 = P3 + F3*S

s_rel = np.arange(0.5, 1.5, 0.01)   # relative changes in pool size
Snew = s_rel*S      # new number of receptors

Wnew1 = 0.5*(Snew+R1+rho1) - np.sqrt(0.25*(Snew+R1+rho1)**2 - R1*Snew)
Wnew2 = 0.5*(Snew+R2+rho2) - np.sqrt(0.25*(Snew+R2+rho2)**2 - R2*Snew)
Wnew3 = 0.5*(Snew+R3+rho3) - np.sqrt(0.25*(Snew+R3+rho3)**2 - R3*Snew)

Fnew1 = Wnew1/Snew
Fnew2 = Wnew2/Snew
Fnew3 = Wnew3/Snew

hetero_change1 = 100.0*(Fnew1-F1)/F1
hetero_change2 = 100.0*(Fnew2-F2)/F2
hetero_change3 = 100.0*(Fnew3-F3)/F3

# plot result     
f = plt.figure(figsize=(4,3))
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 12}
plt.rc('font', **font)
plt.rc('font', serif='Times New Roman') 
plt.plot(100*s_rel, hetero_change1, 'b', label=r'$F = 0.5$')
plt.plot(100*s_rel, hetero_change2, 'k', label=r'$F = 0.7$')
plt.plot(100*s_rel, hetero_change3, 'r', label=r'$F = 0.9$')
#plt.axis((40.0, 160.0, -20.0, 20.0))
plt.xlabel(r'relative number of slots (%)')
plt.ylabel(r'relative efficacy change (%)')
plt.legend(loc='upper right', fontsize=12)
plt.title(r'$\phi = 2.67$')
plt.show()
#f.savefig("Fig5C.pdf", bbox_inches='tight')




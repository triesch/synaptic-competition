#!/usr/bin/env python

# Illustration of multiplicative scaling of synaptic weights
#
# Jochen Triesch, January 2017 - January 2018
#
# This program was used to create Fig. 2B of the manuscript.
# It plots the empirical cumulative distribution function of synaptic weights
# for a set of synapses with lognormally distributed slot numbers for three
# different filling fractions.

import numpy as np
import matplotlib.pyplot as plt

N = 100   # number of synapses
S = 100*N # total number of slots

F1 = 0.5
F2 = 0.7
F3 = 0.9

W1 = F1*S 
W2 = F2*S
W3 = F3*S

print('W1 = ', W1, ', W2 = ', W2, ', W3 = ', W3)

s = np.random.lognormal(1.0, 0.25, N)
s = s/np.sum(s)*S # normalize so that sum equals S
sorted_s = np.sort(s)
      
log_w1 = np.log10(sorted_s*F1)
log_w2 = np.log10(sorted_s*F2)
log_w3 = np.log10(sorted_s*F3)

yvals = np.linspace(0, 1, len(sorted_s)+1)

f = plt.figure(figsize=(4,3))
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 12}
plt.rc('font', **font)
plt.rc('font', serif='Times New Roman') 
plt.step(sorted(list(log_w1) + [max(log_w1)]), yvals, 'b', label=r'$F = 0.5$')
plt.step(sorted(list(log_w2) + [max(log_w2)]), yvals, 'k', label=r'$F = 0.7$')
plt.step(sorted(list(log_w3) + [max(log_w3)]), yvals, 'r', label=r'$F = 0.9$')
plt.xlabel(r'$\log \, w$')
plt.ylabel('Cumulative Probability')
plt.legend(loc='lower right', fontsize=12)
plt.show()
#f.savefig("Fig2B.pdf", bbox_inches='tight')




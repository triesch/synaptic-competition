# Plotting CV from multiple simulations

# This program was used to create Fig. 8C,D of the manuscript.
# It plots the CV of w_i of 7 synapses from multiple simulations and combines them into one figure.
# To get the data for the CV please run stochasticSimulation.py

import numpy as np
from  matplotlib import pyplot as plt
from scipy import stats,polyval, polyfit, linspace

# extract the data from the text files
cv1 = np.array([])
data = open("cv_ff1") # please name the files with CV data differently, like cv_ff1, cv_ff2, ...
counter = 0
for line in data:
    counter += 1
    if counter == 1:
        continue
    if counter <= 9:
        cv1 = np.append(cv1, float(line.rstrip())) 
    if counter == 10:
        break
data.close()

cv2 = np.array([])
data = open("cv_ff2")
counter = 0
for line in data:
    counter += 1
    if counter == 1:
        continue
    if counter <= 9:
        cv2 = np.append(cv2, float(line.rstrip())) 
    if counter == 10:
        break
data.close()

cv3 = np.array([])
data = open("cv_ff3")
counter = 0
for line in data:
    counter += 1
    if counter == 1:
        continue
    if counter <= 9:
        cv3 = np.append(cv3, float(line.rstrip())) 
    if counter == 10:
        break
data.close()

# regression
x_axis1 = np.array([1,2,5,10,20,50,100])
x_axis2 = np.array([1,2,5,10,20,50,100])
x_axis3 = np.array([1,2,5,10,20,50,100])

av_x_axis1 = sum(np.log(x_axis1))/len(x_axis1)
av_x_axis2 = sum(np.log(x_axis2))/len(x_axis2)
av_x_axis3 = sum(np.log(x_axis3))/len(x_axis3)

av_cv1 = sum(np.log(cv1[:-1]))/len(cv1[:-1])
av_cv2 = sum(np.log(cv2[:-1]))/len(cv2[:-1])
av_cv3 = sum(np.log(cv3[:-1]))/len(cv3[:-1])

b_cv1=sum((np.log(x_axis1)-av_x_axis1)*(np.log(cv1[:-1])-av_cv1))/sum((np.log(x_axis1)-av_x_axis1)**2)
a_cv1=av_cv1-b_cv1*av_x_axis1
b_cv2=sum((np.log(x_axis2)-av_x_axis2)*(np.log(cv2[:-1])-av_cv2))/sum((np.log(x_axis2)-av_x_axis2)**2)
a_cv2=av_cv2-b_cv2*av_x_axis2
b_cv3=sum((np.log(x_axis3)-av_x_axis3)*(np.log(cv3[:-1])-av_cv3))/sum((np.log(x_axis3)-av_x_axis3)**2)
a_cv3=av_cv3-b_cv3*av_x_axis3

X = np.linspace(0.8,120,500,endpoint=True)
Y_cv1 = np.exp(a_cv1)*(X**b_cv1)
Y_cv2 = np.exp(a_cv2)*(X**b_cv2)
Y_cv3 = np.exp(a_cv3)*(X**b_cv3)

print("cv1: f1(x)="+str(np.exp(a_cv1))+"*x^("+str(b_cv1)+")")
print("cv2: f2(x)="+str(np.exp(a_cv2))+"*x^("+str(b_cv2)+")")
print("cv3: f3(x)="+str(np.exp(a_cv3))+"*x^("+str(b_cv3)+")")

# plot
f = plt.figure(figsize=(4,3))
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 12}
plt.rc('font', **font)
plt.rc('font', serif='Times New Roman') 
plt.rc('text', usetex=True)

plt.xscale('log')
plt.yscale('log')
plt.plot(x_axis1, cv1[:-1],'.',color='green',label=r'$\phi=1.0$')
plt.plot(X, Y_cv1,'-',color='green',alpha = 0.3)
plt.plot(x_axis2, cv2[:-1],'.',color='cyan',label=r'$\phi=2.67$')
plt.plot(X, Y_cv2,'-',color='cyan',alpha = 0.3)
plt.plot(x_axis3, cv3[:-1],'.',color='darkviolet',label=r'$\phi=5.0$')
plt.plot(X, Y_cv3,'-',color='darkviolet',alpha = 0.3)
plt.ylabel(r'$CV \; [{\rm \%}]$', fontsize=12)
plt.xlabel(r'$s_i$', fontsize=12)
#plt.title(r'$p=2.67$', fontsize=12)
plt.legend(loc=1, fontsize=12)
plt.show()
f.savefig("plot_cv_FigCD.pdf", bbox_inches='tight')

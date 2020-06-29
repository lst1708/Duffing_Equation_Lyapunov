"""Duffing Equation"""
import time 
start=time.time()
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import scipy.signal
plt.close('all')
#Constants for the duffing oscillator 
a=1        #cubic constant
b=1/3   # linear constant
delta=0.2  #damping constant
gamma= 2.5 #force magnitude constant
w=2        #oscillating frequency of driving force
v0=0.1     #starting velocity
x0=0.1    #starting position
startconditions=np.array((x0,v0))#Putting starting conditions into single array

def duffing(X,t):
    x=X[0]
    xvelocity=X[1]
    xacceleration=-delta*xvelocity+a*x-b*x**3+gamma*np.cos(w*t)
    return xvelocity,xacceleration

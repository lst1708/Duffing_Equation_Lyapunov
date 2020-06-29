"""Duffing Equation"""
import time 
start=time.time()
import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt
plt.close("all")
"""Setting initial conditions (remember python starts counting at 0)"""
tlist=[]
tlist.append(0)
tlistsol=[]

tlistsol2=[]
sol2=[]
sol=[]
n=2 #number of non linear equations
nn=6 #number of total equations (linear+2*non-linear)

""" Creating arrays for vectors"""
x=np.zeros((nn))
zvector=np.zeros((n))
GSC=np.zeros((n))
cum=np.zeros((n))
xprime=np.zeros((nn))

#Constants for the duffing oscillator 
a=-1        #linear constant
b=1/4   # cubic constant
delta=0.2  #damping constant
gamma=2.5 #force magnitude constant
w=2.5        #oscillating frequency of driving force
v0=0.1     #starting velocity
x0=0.5    #starting position

z=[]

def duffing(t,X): #duffing oscillator function used for calculating velocity and acceleration
    
    xprime[0]=X[1]
    xprime[1]=-delta*X[1]-a*X[0]-b*X[0]**3+gamma*np.cos(w*t)
    for i in range(0,2):
        #2 copies of linearized equation of motion
        xprime[3+2*i]=-delta*X[3+i*2]-a*X[2+i*2]-3*b*X[2+i*2]*X[0]**2     #xprime acceleration for orthagnormal vectors 
        xprime[2+2*i]=X[3+i*2]                                            #xprime velocity for orthagnormal vectors
    return xprime                                                         #returns matrix of velocitys and accelerations 

    
orbits=100             #number of orbits
dt=0.5                   #time step


"""Setting initial conditions for normalised linear system (remember python starts counting at 0)"""
x[2]=1
x[5]=1

backend='dopri5' #Integration method
r = ode(duffing).set_integrator(backend)
t=0
r.set_initial_value(x,t)
counter=0  
t1=orbits*2*np.pi*w
while r.successful() and tlist[-1]<t1:  #will integrate up until t1 time is reached
    r.integrate(r.t+dt) #step=True means that variable steps are used, with more steps where gradient is close to zero.
    tlist.append(r.t)  #adds results to already created lists  
    x=r.y
    #Normalise first vector
    for j in range(0,n):
        zvector[0]=zvector[0]+(x[n*(j+1)])**2  
    zvector[0]=np.sqrt(zvector[0])
    for j in range(0,n):              
        x[n*(j+1)]=x[n*(j+1)]/zvector[0]  

    #Genereate new orthonormal vectors
    for j in range(1,n):  
        #Generate j-1 Gsc coefficients
        for k in range(0,j): #not j-1 becuase of range function
            GSC[k]=0.0
            for l in range(0,n):
                GSC[k]=GSC[k]+x[n*(l+1)+j]*x[n*(l+1)+k]
        #construct new vector
        for k in range(0,n):
            for l in range(0,j):
                x[n*(k+1)+j]=x[n*(k+1)+j]-GSC[l]*x[n*(k+1)+l]
        
        #Calculate vector norm
        zvector[j]=0.0
        for  k in range(0,n):
            for i in range(0,n):
              zvector[k]=zvector[k]+x[n*(i+1)+k]**2
            zvector[k]=np.sqrt(zvector[k])
        z.append(zvector)
        
        #Normalise the new vector
        for k in range(0,n):
          for i in range(0,n):
            x[n*(i+1)+k]=x[n*(i+1)+k]/zvector[k]
        
        #update running vector magnitudes
        for k in range(0,n):
            cum[k]=cum[k]+np.log(zvector[k])
        for k in range(0,n):
            if k==0:                #seperated out lists for different k values
                tlistsol.append(r.t)
                sol.append(cum[k]/r.t)
            if k==1:
                tlistsol2.append(r.t)
                sol2.append(cum[k]/r.t)


        
tlistsol=np.array((tlistsol))
sol=np.array((sol))
tlistsol2=np.array((tlistsol2))
sol2=np.array((sol2))
z=np.array((z)) 
plt.figure()           
plt.scatter((tlistsol/(2*w*np.pi)),sol,s=0.75)
plt.scatter((tlistsol2/(2*w*np.pi)),sol2,s=0.75)
plt.title("Coveregence of Lyapunov exponents for reference parameters")
plt.xlabel("Number of orbits")
plt.ylabel(r"Lyapunov exponent($s^{-1}$)")
plt.draw()


dimension=1+ (sol[-1]/abs(sol2[-1]))    #Lyupunov Dimesion Calculation

end=time.time()
print("Time taken is "+str(end-start)+" seconds")
print("First exponent is "+str(sol[-1]))
print("Second exponent is "+str(sol2[-1]))
dimension=str(dimension)


print("This is the Lynapunov dimension  " +dimension )
plt.show(block=True)
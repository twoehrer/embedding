#Compute the Lyapunov exponent for simple example x' = e^lambda x


import numpy as np
import matplotlib.pyplot as plt


def evolution(x, t, l = 1):
   return  np.exp(l * t) * x 

def logistic(x_0,r = 3.2, reps = 600):
    x = np.zeros(reps)
    x[0] = x_0
    for i in range(reps - 1):
        x[i + 1] = r * x[i] * (1 - x[i])
    return x

    
    
def traj(x, l = 1):
    traj = np.zeros((100), dtype = np.float64)
    for i in np.arange(100):
        traj[i] = evolution(x,i,l)
    return traj
    
print(traj(1))

def le(x,y):
    d = x - y
    
    d = d/d[0]
    d = np.log(d)
    
    t = np.arange(len(x))
    
    d = d/t
    return d
r = 3.8 #logistic parameter
x = logistic(0.5,r)
y = logistic(0.5001,r)

le = le(x,y)
t = np.arange(600)
plt.plot(t, x,y)
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.animation import FuncAnimation

import matplotlib.pyplot as plt
import numpy as np
import scipy . integrate as sp
from matplotlib import animation

 

def pendule(y,t):
    return y[1],g*np.sin(y[0])*(L*M-l*m)/(L*L*M+l*l*m)

 

def trebuchet(y,t,prec):
    d2_theta  = 1/(m*l*l+M*L*L*np.sin(y[0]-y[2])**2)*(M*L*(R*y[3]*y[3]-L*y[1]*y[1]*np.cos(y[0]-y[2]))*np.sin(y[0]-y[2])+g*(M*L-m*l)*np.sin(y[0])-M*g*L*np.sin(y[2])*np.cos(y[0]-y[2]))
    d2_phi = L/R*np.cos(y[0]-y[2])*d2_theta\
        -L/R*y[1]*y[1]*np.sin(y[0]-y[2])-g/R*np.sin(y[2])
    if d2_phi<prec and y[4]*y[4]+y[6]*y[6]<=(R+L)*(R+L):
        return y[1],d2_theta,\
                y[3],d2_phi,\
                y[5],-L*d2_theta*np.cos(y[0])+L*y[1]*y[1]*np.sin(y[0])+R*d2_phi*np.cos(y[2])-R*y[3]*y[3]*np.sin(y[2]),\
                y[7],-L*d2_theta*np.sin(y[0])-L*y[1]*y[1]*np.cos(y[0])+R*d2_phi*np.sin(y[2])+R*y[3]*y[3]*np.cos(y[2])      

    elif y[6]>-10 :
        return y[1],d2_theta,\
                y[3],d2_phi,\
                y[5],0,\
                y[7],-g

    else :
        
        return y[1],d2_theta,\
                y[3],d2_phi,\
                0,0,\
                0,0

g = 9.81
l = 1
L = 10
m = 18000
M = 200
R = 4

Tf = (np.sqrt((L+R)/g)+2*np.pi*np.sqrt(l/g))*5
print("Tf",Tf)

theta = np.pi/2
phi = 1*np.pi/3
x = R*np.sin(phi)-L*np.sin(theta)
y = -R*np.cos(phi)+L*np.cos(theta)
y0 = [theta,0,phi,0,x,0,y,0]
dt = 2000
t = np.linspace(0,Tf,dt)

l_prec = np.linspace(0,40,5*10)
dist = []
l_distx = []
l_disty = []
for i in range(len(l_prec)):
    ysol = sp.odeint(trebuchet,y0,t,args = (l_prec[i],), rtol=1e-6, atol=1e-8)
    dist.append(ysol[-1,4])
    l_distx.append(ysol[:,4])
    l_disty.append(ysol[:,6])
    print(round((i+1)/len(l_prec)*100,5),r"% finie")

indice_max, valeur_max = max(enumerate(dist), key=lambda x: x[1])
indice_min, valeur_min = min(enumerate(dist), key=lambda x: x[1])
print("distance maximale :", round(valeur_max,2),"m")
print("distance minimale :", round(valeur_min,2),"m")
print("angle de la distance maximale :", round(l_prec[indice_max],1))
print("angle de la distance minimale :", round(l_prec[indice_min],1))

nbr_val = 10
for i in range(nbr_val):
    k = int(len(l_prec)*i/nbr_val)
    print(k,max(l_disty[k]))
    plt.plot(l_distx[k],l_disty[k],label=str(round(l_distx[k][-1]))+"m ")



ysol = sp.odeint(trebuchet,y0,t,args = (l_prec[indice_max],),rtol=1e-6, atol=1e-8)

lx = l*np.sin(ysol[:,0])
ly = -l*np.cos(ysol[:,0])
Lx = -lx*L/l
Ly = -ly*L/l
Rx = R*np.sin(ysol[:,2])
Ry = -R*np.cos(ysol[:,2])
Mx = ysol[:,4]
My = ysol[:,6]

ysol = sp.odeint(trebuchet,y0,t,args = (l_prec[indice_min],),rtol=1e-6, atol=1e-8)

min_lx = l*np.sin(ysol[:,0])
min_ly = -l*np.cos(ysol[:,0])
min_Lx = -lx*L/l
min_Ly = -ly*L/l
min_Rx = R*np.sin(ysol[:,2])
min_Ry = -R*np.cos(ysol[:,2])
min_Mx = ysol[:,4]
min_My = ysol[:,6]

plt.plot(Mx,My,label="max")

plt.legend()
plt.xlabel("x(m)")
plt.ylabel("y(m)")

fig = plt.figure(figsize=(12,10))
plt.xlim(min(min_Mx)*1.1,max(Mx)*1.1)
plt.ylim(-15,max(My)*1.01)
plt.grid()
plt.xlabel("x(m)")
plt.ylabel("y(m)")
plt.title("animation : trebuchet")

line, = plt.plot([],[],"-o")
string, = plt.plot([],[],"-o")
Masse, = plt.plot([],[],"o")
Masse_min, = plt.plot([],[],"o")
sol, = plt.plot([],[],"-")
traj, = plt.plot([],[],"--")
traj_min,= plt.plot([],[],"--")

def init():
    line.set_data([], [])
    string.set_data([],[])
    Masse.set_data([],[])
    Masse_min.set_data([],[])
    sol.set_data([],[])
    traj.set_data([],[])
    traj_min.set_data([],[])
    return line,string,Masse,Masse_min,sol,traj,traj_min,

 

def animate(i):
    line.set_data([Lx[i],lx[i]],[Ly[i],ly[i]])
    string.set_data([Rx[i]+Lx[i],Lx[i]],[Ry[i]+Ly[i],Ly[i]])
    Masse.set_data(Mx[i],My[i])
    Masse_min.set_data(min_Mx[i],min_My[i])
    sol.set_data([-15,max(Mx)*1.1],[-10,-10])
    px = []
    py = []
    px_min = []
    py_min = []
    for k in range(i):
        px.append(Mx[k])
        py.append(My[k])
        px_min.append(min_Mx[k])
        py_min.append(min_My[k])
    traj.set_data([px],[py])
    traj_min.set_data([px_min],[py_min])
    return line,string,Masse,Masse_min,sol,traj,traj_min

ani = animation.FuncAnimation(fig, animate, init_func=init, frames=dt, blit=True, interval=20,repeat=True)

plt.show()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 21:45:05 2023

@author: vedantvairagi

Using Functions for equations
"""
import numpy as np
import matplotlib.pyplot as plt
from linearequations import *
import random


r = 0.25
α = 0.125
γ = 0.75
b0 = 2.5
c = 1
Tnon = 7.5
hnon = 150
tnon = 2
dt = 1/30
tmin = 0
tmax = 41*5
dtnon = dt/tnon
tfin = tmax/tnon
nt = int(tfin/dtnon)

Times=np.linspace(tmin,tfin,nt)

def RungaKutta(Tinit,hinit,µo,µann,τ,t,en,fann,fran,τcorr,E2):
    
    Tfin = []
    hfin = []
    Tfin1 = []
    hfin1 = []

    for i in range(12):
        Tarray = (Tinit/Tnon)*np.ones((nt,1))
        harray = (hinit/hnon)*np.ones((nt,1))
        WT = np.random.uniform(-0.2,0.2,(nt,1))
        Wh = np.random.uniform(-2,2,(nt,1))
        Tnew = Tarray + WT
        hnew = harray + Wh 
        Tfin.append(Tnew)
        hfin.append(hnew)
        
    for j in range(12):
        T = Tfin[j]
        h = hfin[j]
        
        for i in range(nt-1):
        
            W = np.random.uniform(-1,1)
            E1 = noisywind(fann,fran,τ,t[i],W,τcorr,dtnon)
            b = b0*Selfexcite(µo,µann,τ,t[i])
            R = γ*b - c 
        
            k1 = Tval(b,R,γ,T[i],h[i],en,E1,E2)
            l1 = hval(r,α,b,T[i],h[i],E1)
        
            k2 = Tval(b,R,γ,T[i]+(k1*dtnon/2),h[i]+(l1*dtnon/2),en,E1,E2)
            l2 = hval(r,α,b,T[i]+(k1*dtnon/2),h[i]+(l1*dtnon/2),E1)
        
            k3 = Tval(b,R,γ,T[i]+(k2*dtnon/2),h[i]+(l2*dtnon/2),en,E1,E2)
            l3 = hval(r,α,b,T[i]+(k2*dtnon/2),h[i]+(l2*dtnon/2),E1)
    
            k4 = Tval(b,R,γ,T[i]+(k3*dtnon),h[i]+(l3*dtnon),en,E1,E2)
            l4 = hval(r,α,b,T[i]+(k3*dtnon),h[i]+(l3*dtnon),E1)
        
            T[i+1]=T[i]+ 1/6 * dtnon*(k1 + 2*k2 + 2*k3 + k4)
            h[i+1]=h[i] + 1/6 * dtnon*(l1 + 2*l2 + 2*l3 + l4)
                
        Tfin1.append(T) 
        hfin1.append(h)
            
        
    return Tfin1,hfin1

Tplot,hplot = RungaKutta(1.125,0,0.75,0.2,12/tnon,Times,0.1,0.02,0.2,(1/30)/tnon,0)



for i in range(12):
    Tplotvar = Tplot[i] * Tnon
    hplotvar = hplot[i] * hnon

    fig, ax1 = plt.subplots(figsize=(20,10))
    ax1.plot(Times*tnon,Tplotvar, 'k-', label='Temperature')
    plt.ylabel('T(k)')
    plt.legend(loc='lower right')
    ax2=ax1.twinx()
    ax2.plot(Times*tnon,hplotvar, 'r-', label='Height')
    plt.ylabel('h(m)')
    plt.legend(loc='best',bbox_to_anchor=(1,1))
    ax1.xaxis.set_tick_params(labelsize=24)
    ax1.yaxis.set_tick_params(labelsize=24)
    ax2.yaxis.set_tick_params(labelsize=24)
    plt.xlabel('t (month)')
    plt.ylabel('h (m)')
    plt.title('Ensemble %i' %(i+1),fontsize = 40)
    plt.tight_layout()
    plt.show()
    figname='RK-timeseriesEnsemble %i' %(i+1)
    fig.savefig(figname, dpi=300)


    #Against each other
    fig, ax3 = plt.subplots(figsize=(12,10))
    ax3.plot(Tplotvar,hplotvar,'k-', label='Ensemble %i' %(i+1))
    plt.legend(loc='best')
    ax3.xaxis.set_tick_params(labelsize=24)
    ax3.yaxis.set_tick_params(labelsize=24)
    plt.xlabel('T (K)')
    plt.ylabel('h(m)')
    plt.title('Ensemble %i' %(i+1),fontsize = 30)
    plt.show()
    figname1='RK-phaseEnsemble %i' %(i+1)
    fig.savefig(figname1, dpi=300)
    
for i in range(12):
    Tplotvar = Tplot[i] * Tnon
    hplotvar = hplot[i] * hnon
    
    fig = plt.plot(Times*tnon,Tplotvar,label='Ensemble %i' %(i+1))

plt.ylabel('T(k)')
plt.legend(loc='best', bbox_to_anchor=(1,1))
plt.xlabel('t (month)')
plt.title('Ensemble plume',fontsize = 20)
fig(figsize=(20, 10), dpi=300)
plt.tight_layout()
plt.show()






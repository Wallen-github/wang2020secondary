# -*- coding: utf-8 -*-
"""
Created on Sun May  5 10:59:41 2019

@author: H.S.Wang
"""
import Parameters as Pa
import numpy as np
import PeriodicOrbit as PO
import matplotlib.pyplot as plt
import GetX0 as GX0 
import EOMs as EO

m,A,B,IzA,K,s = Pa.Parameters()

def LoopOE(sol,mu = 1):
    
    S = sol.y[0,:]
    S1 = sol.y[2,:]
    theta1 = sol.y[3,:]
    thetaA1 = np.zeros(len(S))
    a = np.zeros(len(S))
    e = np.zeros(len(S))
    
    for i in range(len(S)):
        thetaA1[i] = (K-m*S[i]**2*theta1[i])/(m*S[i]**2+IzA)
        a[i] = mu/(2*mu/S[i]-(S1[i]**2+S[i]**2*(theta1[i]+thetaA1[i])**2))
        e[i] = np.sqrt(1-S[i]**4*(theta1[i]+thetaA1[i])**2/(mu*a[i]))
        
    return a,e,sol.t

def LoopOE3(sol,mu = 1):
    
    S = sol.y[0,:]
    S1 = sol.y[3,:]
    theta1 = sol.y[4,:]
    thetaA1 = np.zeros(len(S))
    a = np.zeros(len(S))
    e = np.zeros(len(S))
    
    for i in range(len(S)):
        thetaA1[i] = (K-m*S[i]**2*theta1[i])/(m*S[i]**2+IzA)
        a1 = 2/S[i]-(S[i]**2*(theta1[i]+thetaA1[i])**2+S1[i]**2)/mu
        a[i] = 1/a1
        e2 = 1-(S[i]*(theta1[i]+thetaA1[i])**2)**2/(mu*a[i])
        e[i] = np.sqrt(e2)
        
    return a,e,sol.t


def AnaOE(a,e,t):
    
    plt.figure()
    plt.title('Eccentricity')
    plt.xlabel('T')
    plt.ylabel('e')
    plt.plot(t,e)
    plt.show()
    
    plt.figure()
    plt.title('axis')
    plt.xlabel('T')
    plt.ylabel('a')
    plt.plot(a)
    plt.show()

def Plotell(a,e):
    
    c = a*e
    b = np.sqrt(a**2-c**2)
    x = np.linspace(-a,a,1000)
    y1 = b*np.sqrt(1-x**2/a**2)
    y2 = -b*np.sqrt(1-x**2/a**2)
    plt.figure()
    plt.plot(x,y1,color='orange')
    plt.plot(x,y2,color='orange')
    plt.show()


if __name__=="__main__":
    
    
    
    x0,T0 = GX0.GetXLS(0.01,0.0,0)
    x1,T1,flag = PO.PeriOrbit_method(T0,x0,PO.YHC,PO.EOM)
    xx,yy,EE,sol = PO.Plot_orbit(x1,2*T1,'red')
    aa = -1/(2*EE)
    ee = np.sqrt(1+2*EE*K**2)
    a,e,t = LoopOE(sol)
    AnaOE(a,e,t)
#    for i in range(len(a)):
#        Plotell(a[i],e[i])
        
#    thetaA = (K-m*x1[0]**2*x1[3])/(m*x0[0]**2+IzA)
#    x2 = np.array([x1[0],x1[1],0.0,x1[2],x1[3],thetaA])
#    _,_,sol = EO.EOMs_3(2*T1,x2)
#    S = sol.y[0,1]
#    The = sol.y[1,1]+sol.y[2,1]
#    S1 = sol.y[3,1]
#    The1 = sol.y[4,1]+sol.y[5,1]
#    xA = S*np.cos(The)
#    yA = S*np.sin(The)
#    xA1 = S1*np.cos(The)-S*np.sin(The)*The1
#    yA1 = S1*np.cos(The)+S*np.cos(The)*The1
#    k = yA1/xA1
#    bb = yA-k*xA
#    LoopOE3(sol)
    
    
    
    
    
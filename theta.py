# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 00:26:33 2019

@author: H.S.Wang
"""
import numpy as np
import GetX0 as GX
import PeriodicOrbit as PO
import matplotlib.pyplot as plt
import Parameters as Pa
from scipy.optimize import fsolve
from myOdeSolver import OdeSolver
import math

K = 0.15

rs = 120
aA = 6*rs
bA = 5*rs
cA = 4*rs
p = 2000
a_bar = (aA*bA*cA)**(1/3)
rp = 1000
a = rp+aA

MB = 4/3*np.pi*rp**3*p
MA = 4/3*np.pi*(aA*bA*cA)*p
m = MA*MB/(MA+MB)**2
miu = MA/(MA+MB)

alpha = a_bar/a
J2 = (aA**2+bA**2-2*cA**2)/(10*a_bar**2)
J22 = (aA**2-bA**2)/(20*a_bar**2)
A = alpha**2*J2
B = 6*alpha**2*J22

IzA = miu*(aA**2+bA**2)/(5*a**2)

s = [0.5926877939,1.322252203]
S0 = 0.8800540903

def f_1(S):
    theta = 0.0
    return S*(K/(m*S**2+IzA))**2-1/S**2-3/2/S**4*(A+B*np.cos(2*theta))

def AnaXLS(a,b,T,index):
    
#    m,A,B,IzA,K,s = Pa.Parameters()
#    S0 = fsolve(f_1,[2.0])[0]
    Izt = IzA+m*S0**2
    a42 = -2*B*Izt/(IzA*S0**5)
    a43 = -2*K/(S0*Izt)
    a31 = K**2*(IzA-3*m*S0**2)/Izt**3+2/S0**3+6*(A+B)/S0**5
    a34 = 2*S0*K*IzA/Izt**2
    
    alpha = -a43*s[index]/(a42+s[index]**2)
    x0 = np.array([a,b,b*s[index]/alpha,-a*alpha*s[index]])

    
    sol = OdeSolver(YHC,[0,T],x0)
    co = ['blue','red']
    #plt.figure()
    plt.plot(sol.y[0,:],sol.y[1,:],color=co[index])
    
    
    #参考椭圆
    x = np.linspace(-a,a,1000)
    y1 = alpha*x0[0]*np.sqrt(1-x**2/x0[0]**2)
    y2 = -alpha*x0[0]*np.sqrt(1-x**2/x0[0]**2)
    plt.plot(x,y1,color='orange')
    plt.plot(x,y2,color='orange')
    plt.show()
    
    return sol

def YHC(t,x):
    
    S0 = fsolve(f_1,[2.0])[0]
    Izt = IzA+m*S0**2
    a42 = -2*B*Izt/(IzA*S0**5)
    a43 = -2*K/(S0*Izt)
    a31 = K**2*(IzA-3*m*S0**2)/Izt**3+2/S0**3+6*(A+B)/S0**5
    a34 = 2*S0*K*IzA/Izt**2
    
    xi = x[0]
    eta = x[1]
    xi1 = x[2]
    eta1 = x[3]
    y = np.zeros(4)
    y[0] = xi1
    y[1] = eta1
    y[2] = a31*xi+a34*eta1
    y[3] = a42*eta+a43*xi1
    
    return y

if __name__=="__main__":
    
    thetamax = []
    
    alpha1 = 6.0/5
    w02 = 3*(1-miu)*(1-alpha1**2)/(S0**3*(1+alpha1**2))
    e = 0.01
    n = np.sqrt((1+miu)/S0**3)
    
    gammaA = 2*w02*e/(w02-n**2)
    
    a = 0.01
    b = 0.0
    T = 10000
    sol = AnaXLS(a,b,T,1)
    
    N = len(sol.y[0,:])
    theta = np.zeros(N)
    for i in range(N):
        theta[i] = math.atan2(sol.y[1,i],sol.y[0,i]+S0)
    
    if max(abs(theta))<np.pi:
        thetamax.append([miu,max(abs(theta)),gammaA])
        

    
        

    
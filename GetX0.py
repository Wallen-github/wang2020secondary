# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 08:56:43 2019

@author: H.S.Wang
"""

import numpy as np
import Parameters as Pa
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from myOdeSolver import OdeSolver


m,A,B,IzA,K,s = Pa.Parameters()


def f_1(S):
    theta = 0.0
    return S*(K/(m*S**2+IzA))**2-1/S**2-3/2/S**4*(A+B*np.cos(2*theta))

def f_2(S):
    theta = np.pi/2
    return S*(K/(m*S**2+IzA))**2-1/S**2-3/2/S**4*(A+B*np.cos(2*theta))

def GetXLS(a,b,index):
    S0 = fsolve(f_1,[1.0])[0]
    Izt = IzA+m*S0**2
    a42 = -2*B*Izt/(IzA*S0**5)
    a43 = -2*K/(S0*Izt)
    a31 = K**2*(IzA-3*m*S0**2)/Izt**3+2/S0**3+6*(A+B)/S0**5
    a34 = 2*S0*K*IzA/Izt**2
    
    alpha = -a43*s[index]/(a42+s[index]**2)#(a31+s1**2)/(a34*s1) 
    x0 = np.array([S0+a,b,b*s[index]/alpha,-a*alpha*s[index]])
    T0 = np.array([np.pi/s[0],np.pi/s[1]])
    
    return x0,T0[index]

def AnaXLS(a,b,T,index):
    
    m,A,B,IzA,K,s = Pa.Parameters()
    S0 = fsolve(f_1,[2.0])[0]
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
    
    index = 0
    dS = 0.02
    T = 40
    x0,T0 = GetXLS(dS,0,index)
    AnaXLS(dS,0.0,T,index)

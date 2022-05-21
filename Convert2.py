# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 21:52:13 2019

@author: H.S.Wang
"""

import numpy as np
from scipy.optimize import fsolve
import Parameters as Pa
import GetX0 as GX0

m,A,B,IzA,K,s = Pa.Parameters()


def ST2ae(S,theta,thetaA,S1,theta1,thetaA1,miu = 1.0):
    
    f = theta+thetaA
    The1 = theta1+thetaA1
    a = 1/(2/S-((S*The1)**2+S1**2)/miu)
    e = np.sqrt(1-(K-IzA*thetaA1)**2/(miu*a))
    
    return a,e,f

def ae2ST(a,e,f,miu = 1.0,thetaA = 0.0):
    
    S = a*(1-e**2)/(1+e*np.cos(f))
    The = f

    V = np.array([-np.sqrt(miu/(a*(1-e**2)))*np.sin(f),np.sqrt(miu/(a*(1-e**2)))*(np.cos(f)+e)])
    
    The1 = (V[1]*np.cos(f)-V[0]*np.sin(f))/S
    S1 = (V[0]*np.cos(f)-V[1]*np.sin(f))/np.cos(2*f)
    
    
    theta = The-thetaA
    thetaA1 = np.sqrt(1/S**3+3/(2*S**5)*(A+B*np.cos(2*theta)))
    theta1 = The1-thetaA1
    
    return S,theta,thetaA,S1,theta1,thetaA1

    
if __name__=="__main__":
    
#    x=np.zeros(6)
#    x[0]=10
#    x[1]=0.0
#    x[2]=0.0
#    x[3]=0.0
#    x[4]=0.0
#    x[5]=np.sqrt(1/x[0]**3+3/(2*x[0]**5)*(A+B*np.cos(2*x[1])))
    x2,T = GX0.GetXLS(0.01,0.0,0)
    x3 = np.zeros(6)
    x3[0] = x2[0]
    x3[1] = x2[1]
    x3[2] = 0.0
    x3[3] = x2[2]
    x3[4] = x2[3]
    x3[5] = np.sqrt(1/x2[0]**3+3/(2*x2[0]**5)*(A+B*np.cos(2*x2[1])))
    a,e,f = ST2ae(x3[0],x3[1],x3[2],x3[3],x3[4],x3[5])
    print('a=',a,'e=',e,'f=',f)
    S,theta,thetaA,S1,theta1,thetaA1 = ae2ST(a,e,f)
    print('S=',S,'theta=',theta,'thetaA=',thetaA,'S1=',S1,'theta1=',theta1,'thetaA1=',thetaA1)
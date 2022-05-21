# -*- coding: utf-8 -*-
"""
Created on Sun May 26 10:25:53 2019

@author: H.S.Wang
"""

import numpy as np
import GetX0 as GX
import PeriodicOrbit as PO
import matplotlib.pyplot as plt
import Parameters as Pa
from scipy.optimize import fsolve


m,A,B,IzA,K,s = Pa.Parameters()


def f_1(S):
    theta = 0.0
    return S*(K/(m*S**2+IzA))**2-1/S**2-3/2/S**4*(A+B*np.cos(2*theta))

def GetX_AB(alpha,beta,index):
    S0 = fsolve(f_1,[1.0])[0]
    Izt = IzA+m*S0**2
    a42 = -2*B*Izt/(IzA*S0**5)
    a43 = -2*K/(S0*Izt)
    a31 = K**2*(IzA-3*m*S0**2)/Izt**3+2/S0**3+6*(A+B)/S0**5
    a34 = 2*S0*K*IzA/Izt**2
    
    alpha1 = -a43*s[0]/(a42+s[0]**2)#(a31+s1**2)/(a34*s1) 
    alpha2 = -a43*s[1]/(a42+s[1]**2)
    a = alpha
    b = alpha2*beta
    a1 = beta*s[1]
    b1 = alpha1*alpha
    x0 = np.array([S0+a,b,a1,b1])
    T0 = np.array([np.pi/s[0],np.pi/s[1]])
    
    return x0,T0[index]


if __name__=="__main__":
    
    AB_stable = []
    AB_unstable = []
    Num = 200
    T = 150
    index = 1
    co = np.array(['black','red'])
    
    for i in range(0,Num):
        for j in range(0,Num):
            alpha = 0+i*(0.02-0)/Num
            beta = 0+j*(0.5-0)/Num
            
            print('alpha = ',alpha,'beta = ',beta)

            x0,T0 = GetX_AB(alpha,beta,index)
            xx0,yy0,E,sol = PO.Plot_orbit(x0,T,co[index])
            theta = sol.y[1]*180/np.pi
            
            if max(abs(theta))>180:
                AB_unstable.append([alpha,beta])
            else:
                AB_stable.append([alpha,beta])    
    
    import pandas as pd
    dfABs = pd.DataFrame(AB_stable)
    dfABu = pd.DataFrame(AB_unstable)
#    dfABs.to_csv(r"E:\learning\硕士\小行星双星系统\2019.4.22Summary\Data\ABs.txt")
#    dfABu.to_csv(r"E:\learning\硕士\小行星双星系统\2019.4.22Summary\Data\ABu.txt")
                
    AB_stable=np.array(AB_stable)
    AB_unstable=np.array(AB_unstable)
    plt.figure()
    plt.scatter(AB_stable[:,0],AB_stable[:,1],color = 'blue')
    plt.scatter(AB_unstable[:,0],AB_unstable[:,1],color = 'red')
    plt.title('AB')
    plt.xlabel('alpha')
    plt.ylabel('beta')
    plt.show()





